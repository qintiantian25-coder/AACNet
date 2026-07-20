"""
AACNet 盲元补完网络 - 测试与定量评估脚本
生成三联对比图（输入|输出|真值）、纯输出图、盲元/闪元指标CSV
"""
import os
import re
import csv
import numpy as np
import torch
import torchvision
from collections import defaultdict
from model import create_model
from dataloader.blind_pixel_loader import create_dataloader
from util.checkpoint_manager import CheckpointManager
from util.metrics import MetricCalculator
import argparse
import sys


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_blind_coords(csv_path):
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
    return np.unique(np.array(coords, dtype=np.int32), axis=0)


def load_flash_map(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return {}
    flash_map = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or 'frame_name' not in reader.fieldnames:
            return {}
        if 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return {}
        for row in reader:
            try:
                fname = os.path.basename(str(row['frame_name']))
                x = int(float(row['x']))
                y = int(float(row['y']))
            except Exception:
                continue
            flash_map.setdefault(fname, set()).add((x, y))
    return {k: list(v) for k, v in flash_map.items()}


class GroupStats:
    def __init__(self):
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0
        self.img_count = 0
        self.blind_abs_sum = 0.0
        self.blind_sq_sum = 0.0
        self.blind_pix_sum = 0
        self.blind_abs_in_sum = 0.0
        self.blind_sq_in_sum = 0.0

    def update(self, psnr, ssim, blind_abs=0.0, blind_sq=0.0, blind_cnt=0,
               blind_abs_in=0.0, blind_sq_in=0.0):
        self.psnr_sum += psnr
        self.ssim_sum += ssim
        self.img_count += 1
        if blind_cnt > 0:
            self.blind_abs_sum += blind_abs
            self.blind_sq_sum += blind_sq
            self.blind_pix_sum += blind_cnt
            self.blind_abs_in_sum += blind_abs_in
            self.blind_sq_in_sum += blind_sq_in

    def get_averages(self):
        n = self.img_count if self.img_count > 0 else 1
        avg_psnr = self.psnr_sum / n
        avg_ssim = self.ssim_sum / n
        result = {'psnr': avg_psnr, 'ssim': avg_ssim, 'img_count': self.img_count,
                  'blind_mae': None, 'blind_rmse': None, 'blind_psnr': None,
                  'blind_mae_input': None, 'blind_count': 0}
        if self.blind_pix_sum > 0:
            mae = self.blind_abs_sum / self.blind_pix_sum
            mse = self.blind_sq_sum / self.blind_pix_sum
            result['blind_mae'] = mae
            result['blind_rmse'] = float(np.sqrt(mse))
            result['blind_psnr'] = float(10.0 * np.log10((255.0 ** 2) / max(mse, 1e-12)))
            result['blind_count'] = int(self.blind_pix_sum)
            if self.blind_abs_in_sum > 0:
                result['blind_mae_input'] = self.blind_abs_in_sum / self.blind_pix_sum
        return result


class FinalReport:
    def __init__(self):
        self.groups = defaultdict(GroupStats)
        self.global_stats = GroupStats()

    def add_image(self, group_name, psnr, ssim, blind_abs=0.0, blind_sq=0.0, blind_cnt=0,
                  blind_abs_in=0.0, blind_sq_in=0.0):
        self.groups[group_name].update(psnr, ssim, blind_abs, blind_sq, blind_cnt,
                                       blind_abs_in, blind_sq_in)
        self.global_stats.update(psnr, ssim, blind_abs, blind_sq, blind_cnt,
                                 blind_abs_in, blind_sq_in)

    def print_result(self):
        sorted_groups = sorted(self.groups.keys(), key=natural_sort_key)
        print("\n" + "=" * 90)
        header = f"{'Group':<12} | {'Imgs':<5} | {'PSNR':<8} | {'SSIM':<8} | {'Blind MAE':<10} | {'Blind PSNR':<10} | {'In MAE':<10}"
        print(header)
        print("-" * 90)
        for g in sorted_groups:
            r = self.groups[g].get_averages()
            b_mae_str = f"{r['blind_mae']:.4f}" if r['blind_mae'] is not None else "-"
            b_psnr_str = f"{r['blind_psnr']:.2f}" if r['blind_psnr'] is not None else "-"
            in_mae_str = f"{r['blind_mae_input']:.4f}" if r.get('blind_mae_input') is not None else "-"
            print(f"{g:<12} | {r['img_count']:<5d} | {r['psnr']:<8.3f} | {r['ssim']:<8.4f} | {b_mae_str:<10} | {b_psnr_str:<10} | {in_mae_str:<10}")
        print("-" * 90)
        gr = self.global_stats.get_averages()
        b_mae_str = f"{gr['blind_mae']:.4f}" if gr['blind_mae'] is not None else "-"
        b_psnr_str = f"{gr['blind_psnr']:.2f}" if gr['blind_psnr'] is not None else "-"
        in_mae_str = f"{gr['blind_mae_input']:.4f}" if gr.get('blind_mae_input') is not None else "-"
        print(f"{'GLOBAL AVG':<12} | {gr['img_count']:<5d} | {gr['psnr']:<8.3f} | {gr['ssim']:<8.4f} | {b_mae_str:<10} | {b_psnr_str:<10} | {in_mae_str:<10}")
        print("=" * 90 + "\n")


def run_test(config, device=None):
    print("\n" + "=" * 60)
    print("AACNet 测试与定量评估")
    print("=" * 60)

    if device is None:
        if torch.cuda.is_available() and len(config.gpu_ids) > 0:
            torch.cuda.set_device(config.gpu_ids[0])
            device = torch.device(f'cuda:{config.gpu_ids[0]}')
        else:
            device = torch.device('cpu')

    # 1. 数据加载器
    print("\n[1/4] 构建测试数据加载器...")
    test_loader = create_dataloader(config, phase='test', shuffle=False)
    print(f"  测试集: {len(test_loader.dataset)} 张图像")

    # 2. 创建输出目录
    save_root = os.path.join(getattr(config, 'results_dir', './results'), 'aacnet_blind_test')
    save_triple = os.path.join(save_root, 'triple_comparison')
    save_pure = os.path.join(save_root, 'test')
    save_eval = os.path.join(save_root, 'blind_eval')
    os.makedirs(save_triple, exist_ok=True)
    os.makedirs(save_pure, exist_ok=True)
    os.makedirs(save_eval, exist_ok=True)

    # 3. 模型加载
    print("\n[2/4] 构建模型并加载权重...")
    config.model = config.model_name
    config.isTrain = False
    if not hasattr(config, 'checkpoints_dir'):
        config.checkpoints_dir = config.checkpoint_dir
    if not hasattr(config, 'name'):
        config.name = getattr(config, 'experiment_name', 'aacnet_blind')

    model = create_model(config)
    if hasattr(model, 'net_G'):
        model.net_G = model.net_G.to(device)
    model.eval()

    ckpt_mgr = CheckpointManager(config, config.best_metric)
    # 测试时优先使用当前实验目录下的模型，不使用训练 resume 的 checkpoint_path
    load_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
    if not os.path.exists(load_path):
        load_path = getattr(config, 'checkpoint_path', '') or os.path.join(config.checkpoint_dir, 'best_model.pt')
    if os.path.exists(load_path):
        print(f"  加载权重: {load_path}")
        ckpt_mgr.load_checkpoint(load_path, model, load_weights_only=True)
    else:
        print(f"  [错误] 未找到权重文件: {load_path}")
        return

    # 4. 推理 + 保存图片 + 统计指标
    print("\n[3/4] 开始推理...")
    metric_calc = MetricCalculator(config.crop_border)
    mask_root = os.path.join(config.data_root, getattr(config, 'test_mask_dir', 'test_mask'))
    report = FinalReport()
    per_image_logs = []

    # 缓存 per-group 的 blind/flash 坐标
    group_mask_cache = {}

    def get_group_masks(group_name):
        if group_name in group_mask_cache:
            return group_mask_cache[group_name]
        gmask_dir = os.path.join(mask_root, group_name)
        blind_csv = os.path.join(gmask_dir, 'blind_pixel_coords.csv')
        if not os.path.exists(blind_csv):
            blind_csv = os.path.join(gmask_dir, 'blind_coords.csv')
        flash_csv = os.path.join(gmask_dir, 'flash_pixel_coords.csv')
        result = {
            'blind_coords': load_blind_coords(blind_csv),
            'flash_map': load_flash_map(flash_csv),
        }
        group_mask_cache[group_name] = result
        return result

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            img_name = batch['name'][0]
            group_name = batch['group'][0] if batch['group'] else 'default'

            # 推理
            input_data = {
                'blur': batch['blur'].to(device),
                'sharp': batch['sharp'].to(device),
                'mask': batch['mask'].to(device),
                'img_path': batch['img_path']
            }
            model.set_input(input_data)
            model.test()

            # 提取张量: [-1, 1] → [0, 1]
            blur_t = (model.img_m[0].cpu() + 1) / 2
            out_t = (model.img_out[0].cpu() + 1) / 2
            gt_t = (model.img_truth[0].cpu() + 1) / 2

            # 裁剪 border
            b = config.crop_border
            if b > 0:
                blur_t = blur_t[:, b:-b, b:-b]
                out_t = out_t[:, b:-b, b:-b]
                gt_t = gt_t[:, b:-b, b:-b]

            # --- 保存三联对比图: [输入 | 输出 | 真值] ---
            triple_dir = os.path.join(save_triple, group_name) if group_name else save_triple
            os.makedirs(triple_dir, exist_ok=True)
            comparison = torch.cat([blur_t, out_t, gt_t], dim=2)  # 水平拼接
            torchvision.utils.save_image(comparison, os.path.join(triple_dir, f"triple_{img_name}"))

            # --- 保存纯输出图 ---
            pure_dir = os.path.join(save_pure, group_name) if group_name else save_pure
            os.makedirs(pure_dir, exist_ok=True)
            torchvision.utils.save_image(out_t, os.path.join(pure_dir, img_name))

            # --- 计算指标 ---
            out_uint8 = (out_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt_uint8 = (gt_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            blur_uint8 = (blur_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            psnr, ssim = metric_calc.calculate_psnr_ssim(out_uint8, gt_uint8)

            # 盲元/闪元专项指标
            h, w = gt_uint8.shape[:2]
            masks = get_group_masks(group_name)
            blind_coords = masks['blind_coords']
            flash_map = masks['flash_map']

            merged_coords = []
            if blind_coords is not None:
                xs, ys = blind_coords[:, 0], blind_coords[:, 1]
                valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                if np.any(valid):
                    merged_coords.extend(zip(xs[valid], ys[valid]))

            frame_flash = flash_map.get(img_name, []) if flash_map else []
            for fx, fy in frame_flash:
                if 0 <= fx < w and 0 <= fy < h:
                    merged_coords.append((fx, fy))

            b_abs_err, b_sq_err, b_cnt = 0.0, 0.0, 0
            b_abs_err_in, b_sq_err_in = 0.0, 0.0
            b_mae_val, b_rmse_val, b_psnr_val = None, None, None

            if merged_coords:
                coords_arr = np.unique(np.array(merged_coords, dtype=np.int32), axis=0)
                if coords_arr.size > 0:
                    x_arr, y_arr = coords_arr[:, 0], coords_arr[:, 1]
                    # 转灰度计算误差
                    out_gray = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2GRAY) if out_uint8.ndim == 3 else out_uint8
                    gt_gray = cv2.cvtColor(gt_uint8, cv2.COLOR_RGB2GRAY) if gt_uint8.ndim == 3 else gt_uint8
                    blur_gray = cv2.cvtColor(blur_uint8, cv2.COLOR_RGB2GRAY) if blur_uint8.ndim == 3 else blur_uint8

                    out_vals = out_gray[y_arr, x_arr].astype(np.float64)
                    gt_vals = gt_gray[y_arr, x_arr].astype(np.float64)
                    blur_vals = blur_gray[y_arr, x_arr].astype(np.float64)

                    err_out = out_vals - gt_vals
                    err_in = blur_vals - gt_vals

                    b_abs_err = float(np.abs(err_out).sum())
                    b_sq_err = float((err_out ** 2).sum())
                    b_cnt = len(err_out)

                    b_abs_err_in = float(np.abs(err_in).sum())
                    b_sq_err_in = float((err_in ** 2).sum())

                    b_mae_val = float(np.abs(err_out).mean())
                    b_mse_val = float((err_out ** 2).mean())
                    b_rmse_val = float(np.sqrt(b_mse_val))
                    b_psnr_val = float(10.0 * np.log10((255.0 ** 2) / max(b_mse_val, 1e-12)))

            # 更新统计
            report.add_image(group_name, psnr, ssim, b_abs_err, b_sq_err, b_cnt,
                             b_abs_err_in, b_sq_err_in)

            # 记录单图日志
            log = {
                'image': img_name,
                'group': group_name,
                'psnr': psnr,
                'ssim': ssim,
                'blind_mae': b_mae_val,
                'blind_rmse': b_rmse_val,
                'blind_psnr': b_psnr_val,
                'blind_count': b_cnt,
            }
            per_image_logs.append(log)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_loader):
                print(f"  进度 [{idx + 1}/{len(test_loader)}] | {img_name} | PSNR: {psnr:.2f} | Blind MAE: {b_mae_val or 0:.4f}")

    # 5. 输出报告
    print("\n[4/4] 生成报告...")
    report.print_result()

    # 保存 per-image CSV
    if per_image_logs:
        per_img_csv = os.path.join(save_eval, 'test_blind_metrics.csv')
        keys = ['image', 'group', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr', 'blind_count']
        with open(per_img_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            writer.writeheader()
            for row in per_image_logs:
                row_clean = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
                writer.writerow(row_clean)
        print(f"\n单图指标已保存: {per_img_csv}")

        # 保存 per-group 汇总 CSV
        group_csv = os.path.join(save_eval, 'test_group_summary.csv')
        summary_keys = ['group_name', 'img_count', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr', 'blind_mae_input', 'blind_count']
        with open(group_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_keys)
            writer.writeheader()
            for g_name in sorted(report.groups.keys(), key=natural_sort_key):
                r = report.groups[g_name].get_averages()
                writer.writerow({
                    'group_name': g_name, 'img_count': r['img_count'],
                    'psnr': round(r['psnr'], 3), 'ssim': round(r['ssim'], 4),
                    'blind_mae': round(r['blind_mae'], 6) if r['blind_mae'] is not None else '',
                    'blind_rmse': round(r['blind_rmse'], 6) if r['blind_rmse'] is not None else '',
                    'blind_psnr': round(r['blind_psnr'], 3) if r['blind_psnr'] is not None else '',
                    'blind_mae_input': round(r['blind_mae_input'], 6) if r.get('blind_mae_input') is not None else '',
                    'blind_count': r['blind_count'],
                })
            gr = report.global_stats.get_averages()
            writer.writerow({
                'group_name': 'GLOBAL_AVERAGE', 'img_count': gr['img_count'],
                'psnr': round(gr['psnr'], 3), 'ssim': round(gr['ssim'], 4),
                'blind_mae': round(gr['blind_mae'], 6) if gr['blind_mae'] is not None else '',
                'blind_rmse': round(gr['blind_rmse'], 6) if gr['blind_rmse'] is not None else '',
                'blind_psnr': round(gr['blind_psnr'], 3) if gr['blind_psnr'] is not None else '',
                'blind_mae_input': round(gr['blind_mae_input'], 6) if gr.get('blind_mae_input') is not None else '',
                'blind_count': gr['blind_count'],
            })
        print(f"分组汇总已保存: {group_csv}")

    print(f"\n三联对比图: {save_triple}")
    print(f"纯输出图:   {save_pure}")
    print("测试完成.")


def main():
    parser = argparse.ArgumentParser(description='AACNet 测试脚本')
    parser.add_argument('--config_path', type=str, default='./experiment.cfg', help='配置文件路径')
    args = parser.parse_args()
    if not os.path.exists(args.config_path):
        print(f"错误: 找不到配置文件 {args.config_path}")
        sys.exit(1)
    from util.config_loader import ConfigLoader
    config = ConfigLoader(args.config_path).get_config()
    run_test(config)


if __name__ == '__main__':
    main()
