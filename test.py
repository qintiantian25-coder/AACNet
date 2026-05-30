"""
AACNet 盲元补完网络 - 定量评估与测试脚本
支持按子目录（Group）统计与全数据集汇总，计算全局指标及盲元/闪元特定区域指标。
"""

import os
import argparse
import re
import csv
import cv2
import numpy as np
import torch
from collections import defaultdict
from model import create_model
from dataloader.blind_pixel_loader import create_dataloader
from util.checkpoint_manager import CheckpointManager
from util.metrics import MetricCalculator
import sys
from datetime import datetime

def natural_sort_key(s):
    """自然排序算法，确保文件名按数字顺序排列"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_blind_coords(csv_path):
    """从静态盲元 CSV 文件中读取坐标"""
    if not csv_path or not os.path.exists(csv_path):
        return None
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return None
        for row in reader:
            try:
                coords.append((int(float(row['x'])), int(float(row['y']))))
            except Exception:
                continue
    if len(coords) == 0:
        return None
    # 去重
    arr = np.unique(np.array(coords, dtype=np.int32), axis=0)
    return arr


def load_flash_map(csv_path):
    """从闪元记录 CSV 文件中读取每张图对应的闪元坐标列表"""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    flash_map = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        # 兼容不同的列名设计
        fn_key = 'frame_name' if 'frame_name' in reader.fieldnames else ('image_name' if 'image_name' in reader.fieldnames else None)
        if fn_key is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return {}
        for row in reader:
            try:
                img_name = row[fn_key]
                x = int(float(row['x']))
                y = int(float(row['y']))
                flash_map[img_name].append((x, y))
            except Exception:
                continue
    return flash_map


class GroupStats:
    """用于统计单个子目录(Group)或全局汇总指标的类"""
    def __init__(self):
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.count = 0
        
        # 盲元特定区域指标统计
        self.blind_abs_error_sum = 0.0
        self.blind_squared_error_sum = 0.0
        self.blind_pixel_count = 0

    def update(self, psnr, ssim, b_mae=None, b_mse=None, b_cnt=0):
        self.psnr_sum += psnr
        self.ssim_sum += ssim
        self.count += 1
        if b_mae is not None and b_cnt > 0:
            self.blind_abs_error_sum += b_mae * b_cnt
            self.blind_squared_error_sum += b_mse * b_cnt
            self.blind_pixel_count += b_cnt

    def get_averages(self):
        avg_psnr = self.psnr_sum / self.count if self.count > 0 else 0.0
        avg_ssim = self.ssim_sum / self.count if self.count > 0 else 0.0
        
        b_mae = self.blind_abs_error_sum / self.blind_pixel_count if self.blind_pixel_count > 0 else 0.0
        b_mse = self.blind_squared_error_sum / self.blind_pixel_count if self.blind_pixel_count > 0 else 0.0
        b_rmse = np.sqrt(b_mse)
        b_psnr = 10.0 * np.log10((255.0 ** 2) / max(b_mse, 1e-12)) if self.blind_pixel_count > 0 else 0.0
        
        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'blind_mae': b_mae,
            'blind_rmse': b_rmse,
            'blind_psnr': b_psnr,
            'blind_count': self.blind_pixel_count,
            'img_count': self.count
        }


class FinalReport:
    """打印和输出最终报告格式的辅助类"""
    def __init__(self):
        self.groups = defaultdict(GroupStats)
        self.global_stats = GroupStats()

    def add_image_metrics(self, group_name, psnr, ssim, b_mae=None, b_mse=None, b_cnt=0):
        self.groups[group_name].update(psnr, ssim, b_mae, b_mse, b_cnt)
        self.global_stats.update(psnr, ssim, b_mae, b_mse, b_cnt)

    def print_final_result(self):
        sorted_groups = sorted(self.groups.keys(), key=natural_sort_key)
        
        print("\n" + "="*80)
        print(f"{'Group Name':<25} | {'ImgCount':<8} | {'PSNR':<8} | {'SSIM':<8} | {'Blind MAE':<10} | {'Blind PSNR':<10}")
        print("-" * 80)
        
        for g in sorted_groups:
            res = self.groups[g].get_averages()
            print(f"{g:<25} | {res['img_count']:<8d} | {res['psnr']:<8.3f} | {res['ssim']:<8.4f} | {res['blind_mae']:<10.4f} | {res['blind_psnr']:<10.2f}")
            
        print("-" * 80)
        g_res = self.global_stats.get_averages()
        print(f"{'GLOBAL AVERAGE':<25} | {g_res['img_count']:<8d} | {g_res['psnr']:<8.3f} | {g_res['ssim']:<8.4f} | {g_res['blind_mae']:<10.4f} | {g_res['blind_psnr']:<10.2f}")
        print("="*80 + "\n")


def run_test(config, device=None):
    """
    测试与定量评估核心函数
    
    Args:
        config: 配置对象
        device: 指定计算设备，若无则内部自动创建
    """
    print("\n" + "="*60)
    print("开始定量评估与测试 (Testing & Evaluation)")
    print("="*60)
    
    if device is None:
        if torch.cuda.is_available() and len(config.gpu_ids) > 0:
            torch.cuda.set_device(config.gpu_ids[0])
            device = torch.device(f'cuda:{config.gpu_ids[0]}')
        else:
            device = torch.device('cpu')

    # 1. 初始化测试集数据加载器
    print("\n正在构建测试集数据加载器...")
    test_loader = create_dataloader(config, phase='test', shuffle=False)
    print(f"  测试集大小: {len(test_loader.dataset)} 张图像")
    
    # 2. 从配置关联的 mask 目录加载盲元和闪元坐标 CSV
    mask_dir = os.path.join(config.data_root, getattr(config, 'test_mask_dir', 'test_mask'))
    static_csv = os.path.join(mask_dir, 'blind_pixel_coords.csv')
    flash_csv = os.path.join(mask_dir, 'flash_pixel_coords.csv')
    
    static_coords = load_blind_coords(static_csv)
    flash_map = load_flash_map(flash_csv)
    
    if static_coords is not None:
        print(f"  [成功] 载入静态盲元坐标表: {static_csv}, 共 {len(static_coords)} 个特征点。")
    else:
        print(f"  [提示] 未发现或未能解析静态盲元坐标 CSV: {static_csv}，将完全依赖单图 Mask 的 0 值区域进行区域评估。")
        
    if len(flash_map) > 0:
        print(f"  [成功] 载入动态闪元历史表: {flash_csv}, 涵盖 {len(flash_map)} 帧的闪烁点数据。")

    # 3. 构建并载入模型权重
    print("\n正在构建模型并载入最佳权重...")
    config.model = config.model_name
    config.isTrain = False
    
    # 防御性补全模型基类所需的环境变量
    if not hasattr(config, 'checkpoints_dir'):
        config.checkpoints_dir = config.checkpoint_dir
    if not hasattr(config, 'name'):
        config.name = getattr(config, 'experiment_name', 'aacnet_blind')
        
    model = create_model(config)
    
    if hasattr(model, 'net_G'):
        model.net_G = model.net_G.to(device)
    model.eval()

    checkpoint_manager = CheckpointManager(config, config.best_metric)
    
    # 优先载入用户指定的权重，否则载入模型目录下的 best_model.pt
    if getattr(config, 'checkpoint_path', None) and os.path.exists(config.checkpoint_path):
        load_path = config.checkpoint_path
    else:
        load_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
        
    if os.path.exists(load_path):
        print(f"  载入目标权重文件: {load_path}")
        checkpoint_manager.load_checkpoint(load_path, model, load_weights_only=True)
    else:
        print(f"  [错误] 未找到任何有效的模型权重权重: {load_path}，无法开始测试！")
        return

    # 4. 创建指标计算器和结果保存路径
    metric_calc = MetricCalculator(config.crop_border)
    save_blind_dir = getattr(config, 'results_dir', './experiments/results')
    os.makedirs(save_blind_dir, exist_ok=True)
    
    report = FinalReport()
    per_image_logs = []
    
    # 用于全局扁平化累加的盲元误差计数器
    blind_pix_sum = 0
    blind_abs_sum = 0.0
    blind_sq_sum = 0.0

    print("\n全面启动前向推理与精细化指标计算...")
    print("-" * 60)

    for idx, batch in enumerate(test_loader):
        # 创建标准的输入形态，保持 mask 为原生的单通道状态 [B, 1, H, W]
        input_data = {
            'blur': batch['blur'].to(device),
            'sharp': batch['sharp'].to(device),
            'mask': batch['mask'].to(device),
            'img_path': batch['img_path']
        }
        
        # 基础模型执行前向推理
        model.set_input(input_data)
        model.test()
        
        # 提取网络生成输出与真实参考答案
        output = model.img_out        # 范围 [-1, 1]
        target = model.img_truth      # 范围 [-1, 1]
        
        # 反归一化转换为标准图像范围 [0, 1] 并转为 HWC 维度
        output_np = ((output[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
        target_np = ((target[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
        
        # 裁剪边缘像素
        if config.crop_border > 0:
            b = config.crop_border
            output_np = output_np[b:-b, b:-b, :]
            target_np = target_np[b:-b, b:-b, :]
            
        # 安全截断，映射到标准 uint8 矩阵
        output_uint8 = (np.clip(output_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        target_uint8 = (np.clip(target_np, 0.0, 1.0) * 255.0).astype(np.uint8)
        
        # 4.1 计算基础的全局 PSNR 和 SSIM
        psnr, ssim = metric_calc.calculate_psnr_ssim(output_uint8, target_uint8)
        
        # 4.2 提取元数据信息
        img_path = batch['img_path'][0]
        img_name = batch['name'][0]
        group_name = batch['group'][0] if ('group' in batch and batch['group']) else 'default'
        
        # 4.3 动态构建该图专属的盲元/闪元评估二值掩码坐标点集
        h, w = target_uint8.shape[0], target_uint8.shape[1]
        eval_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 策略A：融入预载入的静态盲元
        if static_coords is not None:
            for pt in static_coords:
                x, y = pt[0], pt[1]
                if 0 <= y < h and 0 <= x < w:
                    eval_mask[y, x] = 1
                    
        # 策略B：融入该图记录的动态闪元
        if img_name in flash_map:
            for pt in flash_map[img_name]:
                x, y = pt[0], pt[1]
                if 0 <= y < h and 0 <= x < w:
                    eval_mask[y, x] = 1
                    
        # 策略C：保底防御，若 CSV 未提供任何坐标，直接提取当前图像独享的单通道 mask 中的 0 遮罩区
        if np.sum(eval_mask) == 0:
            mask_single = batch['mask'][0, 0].cpu().numpy()
            if config.crop_border > 0:
                b = config.crop_border
                mask_single = mask_single[b:-b, b:-b]
            eval_mask = (mask_single < 0.5).astype(np.uint8)

        # 4.4 针对评估掩码覆盖的特定局部盲元盲点计算聚焦指标
        blind_pts = np.where(eval_mask > 0)
        blind_cnt = len(blind_pts[0])
        
        b_mae, b_mse, b_psnr = 0.0, 0.0, 0.0
        if blind_cnt > 0:
            # 统一提取单通道灰度进行均方误差与绝对误差统计
            out_gray = cv2.cvtColor(output_uint8, cv2.COLOR_RGB2GRAY) if len(output_uint8.shape) == 3 else output_uint8
            tgt_gray = cv2.cvtColor(target_uint8, cv2.COLOR_RGB2GRAY) if len(target_uint8.shape) == 3 else target_uint8
            
            diff_abs = np.abs(out_gray[blind_pts].astype(np.float32) - tgt_gray[blind_pts].astype(np.float32))
            b_mae = float(np.mean(diff_abs))
            b_mse = float(np.mean(diff_abs ** 2))
            b_psnr = 10.0 * np.log10((255.0 ** 2) / max(b_mse, 1e-12))
            
            # 全局累加
            blind_pix_sum += blind_cnt
            blind_abs_sum += np.sum(diff_abs)
            blind_sq_sum += np.sum(diff_abs ** 2)
        
        # 更新至分组与全局统计器
        report.add_image_metrics(group_name, psnr, ssim, b_mae, b_mse, blind_cnt)
        
        # 记录单张图片的详尽对齐日志
        per_image_logs.append({
            'image': img_name,
            'group': group_name,
            'psnr': psnr,
            'ssim': ssim,
            'blind_mae': b_mae,
            'blind_rmse': np.sqrt(b_mse),
            'blind_psnr': b_psnr,
            'blind_count': blind_cnt
        })
        
        # 定期在控制台输出单张推理进度
        if (idx + 1) % config.print_freq == 0 or (idx + 1) == len(test_loader):
            print(f" Progress [{idx+1}/{len(test_loader)}] | Img: {img_name} | PSNR: {psnr:.2f} | Blind MAE: {b_mae:.3f}")

    # 5. 分层级保存分组及全数据集汇总报告
    print('\n' + '='*60)
    print('全数据集测试完成汇总报告:')
    print('='*60)
    report.print_final_result()
    
    # 导出总体的全局表格 CSV 报告
    if len(per_image_logs) > 0:
        global_csv = os.path.join(save_blind_dir, 'test_blind_metrics.csv')
        keys = ['image', 'group', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr', 'blind_count']
        with open(global_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(per_image_logs)
        print(f'\n[成功] 已保存所有单张图片的详细评估指标至: {global_csv}')

        # 另外将每个分组的平均指标单独汇总导出一个简要报告表格
        group_summary_csv = os.path.join(save_blind_dir, 'test_group_summary.csv')
        with open(group_summary_csv, 'w', encoding='utf-8', newline='') as f:
            summary_keys = ['group_name', 'img_count', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr', 'blind_count']
            writer = csv.DictWriter(f, fieldnames=summary_keys)
            writer.writeheader()
            
            # 写入每个子组
            for g_name, stats in sorted(report.groups.items(), key=lambda x: natural_sort_key(x[0])):
                res = stats.get_averages()
                writer.writerow({
                    'group_name': g_name, 'img_count': res['img_count'], 'psnr': f"{res['psnr']:.3f}",
                    'ssim': f"{res['ssim']:.4f}", 'blind_mae': f"{res['blind_mae']:.4f}",
                    'blind_rmse': f"{res['blind_rmse']:.4f}", 'blind_psnr': f"{res['blind_psnr']:.3f}",
                    'blind_count': res['blind_count']
                })
            # 写入全局平均
            g_res = report.global_stats.get_averages()
            writer.writerow({
                'group_name': 'GLOBAL_AVERAGE', 'img_count': g_res['img_count'], 'psnr': f"{g_res['psnr']:.3f}",
                'ssim': f"{g_res['ssim']:.4f}", 'blind_mae': f"{g_res['blind_mae']:.4f}",
                'blind_rmse': f"{g_res['blind_rmse']:.4f}", 'blind_psnr': f"{g_res['blind_psnr']:.3f}",
                'blind_count': g_res['blind_count']
            })
        print(f'[成功] 已保存分组聚合汇总报告至: {group_summary_csv}')


def main():
    """独立测试脚本的脚本入口"""
    parser = argparse.ArgumentParser(description='AACNet 独立定量评估脚本')
    parser.add_argument('--config_path', type=str, default='./experiment.cfg', help='配置文件路径')
    args = parser.parse_args()
    
    if not os.path.exists(args.config_path):
        print(f"错误: 找不到配置文件 {args.config_path}")
        sys.exit(1)
        
    from util.config_loader import ConfigLoader
    config_loader = ConfigLoader(args.config_path)
    config = config_loader.get_config()
    
    run_test(config)


if __name__ == '__main__':
    main()
