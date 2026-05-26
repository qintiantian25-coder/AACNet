import os
import argparse
import re
import csv
import cv2
import numpy as np
import torch
from collections import defaultdict
from options import test_options
from model import create_model
import sys

# ======================================================================
# 工具函数
# ======================================================================
def natural_sort_key(s):
    """自然排序键"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_blind_coords(csv_path):
    """加载静态盲元坐标 CSV，返回 Nx2 numpy 数组（x, y）"""
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
    arr = np.unique(np.array(coords, dtype=np.int32), axis=0)
    return arr


def load_flash_map(csv_path):
    """加载闪光盲元 CSV（带 frame_name 列），返回 {frame_name: [(x,y),...]}"""
    if not csv_path or not os.path.exists(csv_path):
        return {}
    flash_map = {}
    with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        if 'frame_name' not in reader.fieldnames or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
            return {}
        for row in reader:
            try:
                fname = os.path.basename(str(row['frame_name']))
                x = int(float(row['x']))
                y = int(float(row['y']))
            except Exception:
                continue
            flash_map.setdefault(fname, set()).add((x, y))
    normalized = {k: list(v) for k, v in flash_map.items()}
    return normalized


def resolve_csv_path(csv_path, data_root):
    """解析 mask 路径（可能是目录或文件）"""
    if not csv_path:
        return None
    if os.path.isdir(csv_path):
        return csv_path
    if os.path.isabs(csv_path) and os.path.exists(csv_path):
        return csv_path
    if os.path.exists(csv_path):
        return csv_path
    if data_root:
        candidate = os.path.join(data_root, csv_path)
        if os.path.exists(candidate):
            return candidate
    return csv_path


def resolve_group_mask_paths(mask_base_path, data_root, group_name):
    """根据组名（如 001）获取该组对应的 blind_coords.csv 和 flash CSV 路径"""
    candidates = []
    if mask_base_path:
        if os.path.isdir(mask_base_path):
            group_dir = os.path.join(mask_base_path, group_name)
            blind_csv_candidates = [
                os.path.join(group_dir, 'blind_coords.csv'),
                os.path.join(group_dir, 'blind_pixel_coords.csv'),
            ]
            flash_csv = os.path.join(group_dir, 'flash_pixel_coords.csv')
        else:
            candidates.append(mask_base_path)
            if data_root and not os.path.isabs(mask_base_path):
                candidates.append(os.path.join(data_root, mask_base_path))
            base_dir = os.path.dirname(mask_base_path)
            if base_dir:
                candidates.append(os.path.join(base_dir, group_name, os.path.basename(mask_base_path)))
            blind_csv_candidates = candidates
            flash_csv = os.path.join(base_dir, group_name, 'flash_pixel_coords.csv') if base_dir else None
    else:
        blind_csv_candidates = []
        flash_csv = None

    if data_root:
        default_group_dir = os.path.join(data_root, 'test_mask', group_name)
        blind_csv_candidates.extend([
            os.path.join(default_group_dir, 'blind_coords.csv'),
            os.path.join(default_group_dir, 'blind_pixel_coords.csv'),
        ])
        if flash_csv is None:
            flash_csv = os.path.join(default_group_dir, 'flash_pixel_coords.csv')

    seen = set()
    blind_csv = None
    for candidate in blind_csv_candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            blind_csv = candidate
            break

    if flash_csv and not os.path.exists(flash_csv):
        flash_csv = None

    return {'blind_csv': blind_csv, 'flash_csv': flash_csv}


def get_group_name(rel_path):
    """从相对路径中提取组名（第一级目录）"""
    parts = os.path.normpath(rel_path).split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return 'root'


def load_mask_image_or_create(mask_path, img_shape):
    """
    加载mask图像或从盲元坐标创建mask
    返回 [0,1] 范围的mask，其中 1 表示有效区域，0 表示盲元区域
    """
    if mask_path and os.path.exists(mask_path):
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is not None:
            # 假设 mask 是二值图，0 为盲元，255 为有效
            mask = (mask_img > 127).astype(np.float32)
            if mask.shape != img_shape:
                mask = cv2.resize(mask, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_NEAREST)
            return mask
    
    # 如果没有mask图像，创建全 1 的mask（表示全部有效）
    return np.ones(img_shape, dtype=np.float32)


def create_mask_from_coords(blind_coords, img_shape):
    """
    从盲元坐标创建mask
    返回 [0,1] 范围的mask，其中 1 表示有效区域，0 表示盲元区域
    """
    mask = np.ones(img_shape, dtype=np.float32)
    if blind_coords is not None and len(blind_coords) > 0:
        h, w = img_shape
        xs = blind_coords[:, 0]
        ys = blind_coords[:, 1]
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        mask[ys[valid], xs[valid]] = 0
    return mask


# ======================================================================
# 图像预处理/后处理
# ======================================================================
def to_tensor_rgb(img, norm=True):
    """
    将 RGB 图（H x W x 3, uint8）转换为模型输入的 tensor。
    返回 (3, H, W) float32 [0,1] 或 [-1,1]
    """
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    
    # HWC -> CHW
    img = img.transpose(2, 0, 1)
    
    # 标准化到 [-1, 1]（如果需要）
    if norm:
        img = img * 2 - 1
    
    return img


def tensor_to_uint8_rgb(out_tensor, denorm=True):
    """
    将模型输出的三通道 tensor [-1,1] 或 [0,1] 转换为 uint8 RGB 图
    """
    out_np = out_tensor.squeeze(0).detach().cpu().numpy()
    
    # CHW -> HWC
    if out_np.ndim == 3:
        out_np = out_np.transpose(1, 2, 0)
    
    # 反标准化
    if denorm:
        out_np = (out_np + 1) / 2
    
    # 确保值在 [0,1] 范围内
    out_np = np.clip(out_np, 0, 1)
    return (out_np * 255).round().astype(np.uint8)


# ======================================================================
# 评估报告类
# ======================================================================
class TestReport:
    def __init__(self, crop_border=0):
        self.crop_border = crop_border
        self.total_psnr = []
        self.total_ssim = []

    def update_metric(self, gt_img, out_img, img_name=None):
        """计算 PSNR 和 SSIM"""
        psnr = self.calculate_psnr(out_img, gt_img, crop_border=self.crop_border)
        ssim = self.calculate_ssim(out_img, gt_img, crop_border=self.crop_border)
        self.total_psnr.append(float(psnr))
        self.total_ssim.append(float(ssim))

    @staticmethod
    def calculate_psnr(img1, img2, crop_border=0):
        """计算 PSNR"""
        if crop_border > 0:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
        
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse < 1e-10:
            return 100
        return 10 * np.log10(255.0 * 255.0 / mse)

    @staticmethod
    def calculate_ssim(img1, img2, crop_border=0):
        """简化的 SSIM 计算"""
        if crop_border > 0:
            img1 = img1[crop_border:-crop_border, crop_border:-crop_border]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border]
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        C1 = 6.5025
        C2 = 58.5225
        
        mean1 = cv2.blur(img1, (11, 11))
        mean2 = cv2.blur(img2, (11, 11))
        mean1_sq = mean1 ** 2
        mean2_sq = mean2 ** 2
        mean1_mean2 = mean1 * mean2
        
        sigma1_sq = cv2.blur(img1 ** 2, (11, 11)) - mean1_sq
        sigma2_sq = cv2.blur(img2 ** 2, (11, 11)) - mean2_sq
        sigma12 = cv2.blur(img1 * img2, (11, 11)) - mean1_mean2
        
        ssim_map = ((2 * mean1_mean2 + C1) * (2 * sigma12 + C2)) / (
            (mean1_sq + mean2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)

    def print_final_result(self):
        if len(self.total_psnr) == 0:
            print('No valid images were evaluated.')
            return
        print(f'Average PSNR: {np.mean(self.total_psnr):.4f} dB')
        print(f'Average SSIM: {np.mean(self.total_ssim):.4f}')


# ======================================================================
# 主测试函数
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='AACNet盲元补完测试脚本')
    parser.add_argument('--data_root', type=str, required=True, 
                        help='数据集根目录，包含 test_sharp, test_blur, test_mask 等')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='模型权重文件 (.pth)')
    parser.add_argument('--save_dir', type=str, default='results/aacnet_blind_test', 
                        help='输出根目录')
    parser.add_argument('--device', type=str, default='cuda', 
                        help='cuda / cpu')
    parser.add_argument('--test_mask_csv', type=str, default=None, 
                        help='盲元 CSV 路径或 test_mask 目录')
    parser.add_argument('--image_border', type=int, default=0, 
                        help='PSNR/SSIM 计算时裁剪边界')
    parser.add_argument('--model', type=str, default='aacnet', 
                        help='模型名称，与模型文件 *_model.py 对应')
    parser.add_argument('--name', type=str, default='aacnet_blind', 
                        help='模型实例名称')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', 
                        help='模型检查点目录')
    parser.add_argument('--gpu_ids', type=str, default='0', 
                        help='GPU IDs')
    
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    save_pure = os.path.join(args.save_dir, 'test')
    save_blind_dir = os.path.join(args.save_dir, 'blind_eval')
    os.makedirs(save_pure, exist_ok=True)
    os.makedirs(save_blind_dir, exist_ok=True)

    print(f'Device: {device}')
    print(f'Data root: {args.data_root}')
    print(f'Checkpoint: {args.checkpoint}')

    # 创建测试选项对象
    opt = test_options.TestOptions()
    opt.parser.set_defaults(
        model=args.model,
        name=args.name,
        checkpoints_dir=args.checkpoints_dir,
        gpu_ids=args.gpu_ids,
        which_iter='latest',
        img_file=os.path.join(args.data_root, 'test_blur'),
        mask_file=os.path.join(args.data_root, 'test_mask'),
        batchSize=1,
        no_augment=True,
        no_flip=True,
        no_rotation=True,
        phase='test',
    )
    opt = opt.parse()

    # 如果提供了checkpoint，覆盖which_iter
    if args.checkpoint:
        # 从文件名中提取模型迭代数
        ckpt_name = os.path.basename(args.checkpoint)
        if ckpt_name.startswith('net_G_'):
            which_iter = ckpt_name.replace('net_G_', '').replace('.pth', '')
            opt.which_iter = which_iter
        # 直接加载checkpoint路径
        opt.checkpoint_path = args.checkpoint

    # 创建模型
    model = create_model(opt)
    model.eval()

    # 如果提供了checkpoint路径，手动加载
    if args.checkpoint and os.path.exists(args.checkpoint):
        try:
            state_dict = torch.load(args.checkpoint, map_location='cpu')
            if isinstance(state_dict, dict):
                # 兼容不同的checkpoint格式
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                # 去除 'module.' 前缀
                new_state = {}
                for k, v in state_dict.items():
                    new_k = k[7:] if k.startswith('module.') else k
                    new_state[new_k] = v
                state_dict = new_state
            model.net_G.load_state_dict(state_dict)
            print(f'Successfully loaded checkpoint from {args.checkpoint}')
        except Exception as e:
            print(f'Warning: Failed to load checkpoint: {e}')
            print('Will continue with random initialized model')

    # 构建 GT 映射
    gt_root = os.path.join(args.data_root, 'test_sharp')
    gt_map = {}
    for root, _, files in os.walk(gt_root):
        for f in files:
            if f.lower().endswith('.png'):
                full = os.path.join(root, f)
                rel = os.path.normpath(os.path.relpath(full, gt_root))
                gt_map[rel] = full
                if f not in gt_map:
                    gt_map[f] = full

    # 收集输入图像（test_blur）
    input_root = os.path.join(args.data_root, 'test_blur')
    input_files = []
    for root, _, files in os.walk(input_root):
        for f in files:
            if f.lower().endswith('.png'):
                input_files.append(os.path.join(root, f))
    input_files = sorted(input_files, key=natural_sort_key)

    if len(input_files) == 0:
        print(f'No test images found in {input_root}')
        return

    # 按组分组
    grouped_inputs = defaultdict(list)
    for in_path in input_files:
        rel_in = os.path.normpath(os.path.relpath(in_path, input_root))
        grouped_inputs[get_group_name(rel_in)].append(in_path)

    # 解析 mask 根路径
    resolved_test_mask_csv = resolve_csv_path(args.test_mask_csv, args.data_root)

    # 全局统计
    report = TestReport(crop_border=args.image_border)
    blind_abs_sum = 0.0
    blind_sq_sum = 0.0
    blind_abs_in_sum = 0.0
    blind_sq_in_sum = 0.0
    blind_pix_sum = 0
    per_image_logs = []

    print(f'===> 开始定量打分，准备比对 {len(input_files)} 张图片...')
    with torch.no_grad():
        for group_name, group_files in grouped_inputs.items():
            print(f'\n===> Processing group {group_name} ({len(group_files)} images) ...')
            group_rows = []
            group_pure_dir = os.path.join(save_pure, group_name)
            os.makedirs(group_pure_dir, exist_ok=True)

            # 获取该组的盲元 CSV
            masks = resolve_group_mask_paths(resolved_test_mask_csv, args.data_root, group_name)
            blind_coords = load_blind_coords(masks['blind_csv']) if masks['blind_csv'] else None
            flash_map = load_flash_map(masks['flash_csv']) if masks['flash_csv'] else {}

            if masks['blind_csv'] and blind_coords is not None:
                print(f"Loaded blind coords for group {group_name} from: {masks['blind_csv']} ({len(blind_coords)} unique points)")
            elif masks['blind_csv']:
                print(f"WARN: blind coords CSV not loaded for group {group_name}: {masks['blind_csv']}")
            else:
                print(f'WARN: no blind coords CSV found for group {group_name}')

            if masks['flash_csv'] and len(flash_map) > 0:
                print(f"Loaded flash coords map for group {group_name} from: {masks['flash_csv']} ({len(flash_map)} frames)")
            elif masks['flash_csv']:
                print(f"WARN: flash CSV has no valid frame_name/x/y entries for group {group_name}: {masks['flash_csv']}")

            for idx, in_path in enumerate(group_files):
                name = os.path.basename(in_path)
                rel_in = os.path.normpath(os.path.relpath(in_path, input_root))

                # 读取输入图像（RGB）
                in_img = cv2.imread(in_path)
                if in_img is None:
                    print(f'WARN: failed to load {in_path}')
                    continue
                
                # BGR -> RGB
                in_img_rgb = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
                h, w = in_img_rgb.shape[:2]

                # 创建或加载 mask
                mask_path = os.path.join(args.data_root, 'test_mask', group_name, name.replace('.png', '.png'))
                mask = load_mask_image_or_create(mask_path, (h, w))
                
                # 如果有盲元坐标，优先使用坐标生成mask
                if blind_coords is not None and len(blind_coords) > 0:
                    mask = create_mask_from_coords(blind_coords, (h, w))

                # 预处理
                inp_np = to_tensor_rgb(in_img_rgb, norm=True)
                inp_tensor = torch.from_numpy(inp_np).float().unsqueeze(0).to(device)
                
                mask_np = mask
                mask_tensor = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(device)
                # 扩展mask到 RGB 三通道
                mask_tensor = mask_tensor.repeat(1, 3, 1, 1)

                # 创建输入字典
                input_data = {
                    'img': inp_tensor,
                    'mask': mask_tensor,
                    'img_path': [name]
                }

                # 模型推理
                model.set_input(input_data)
                model.test()

                # 获取输出
                out_tensor = model.img_out  # [-1, 1]
                out_img_rgb = tensor_to_uint8_rgb(out_tensor, denorm=True)

                # 查找对应的 GT
                gt_path = gt_map.get(rel_in, gt_map.get(name))
                if gt_path and os.path.exists(gt_path):
                    gt_img_bgr = cv2.imread(gt_path)
                    if gt_img_bgr is None:
                        print(f'WARN: failed to load gt for {name}')
                        # 仍然保存输出图像
                        out_img_bgr = cv2.cvtColor(out_img_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(group_pure_dir, name), out_img_bgr)
                        continue

                    gt_img_rgb = cv2.cvtColor(gt_img_bgr, cv2.COLOR_BGR2RGB)

                    # 尺寸对齐
                    if out_img_rgb.shape != gt_img_rgb.shape:
                        out_img_rgb = cv2.resize(out_img_rgb, (gt_img_rgb.shape[1], gt_img_rgb.shape[0]))
                    if in_img_rgb.shape != gt_img_rgb.shape:
                        in_img_resized = cv2.resize(in_img_rgb, (gt_img_rgb.shape[1], gt_img_rgb.shape[0]))
                    else:
                        in_img_resized = in_img_rgb

                    # 保存输出
                    out_img_bgr = cv2.cvtColor(out_img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(group_pure_dir, name), out_img_bgr)

                    # 转换为灰度用于计算指标
                    gt_gray = cv2.cvtColor(gt_img_rgb, cv2.COLOR_RGB2GRAY)
                    out_gray = cv2.cvtColor(out_img_rgb, cv2.COLOR_RGB2GRAY)
                    in_gray = cv2.cvtColor(in_img_resized, cv2.COLOR_RGB2GRAY)

                    # 全图指标
                    report.update_metric(gt_gray, out_gray, name)
                    full_psnr = report.total_psnr[-1]
                    full_ssim = report.total_ssim[-1]

                    # 盲元评估
                    row = {
                        'image': name,
                        'psnr': full_psnr,
                        'ssim': full_ssim,
                        'blind_mae': None,
                        'blind_rmse': None,
                        'blind_psnr': None,
                        'blind_mae_input': None,
                        'blind_mae_gain_abs': None,
                        'blind_mae_gain_pct': None,
                        'blind_count': 0
                    }

                    # 合并所有盲元坐标（静态 + 帧级闪光）
                    merged_coords = []
                    if blind_coords is not None and blind_coords.size > 0:
                        xs = blind_coords[:, 0]
                        ys = blind_coords[:, 1]
                        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                        if np.any(valid):
                            merged_coords.extend(zip(xs[valid].tolist(), ys[valid].tolist()))

                    if len(flash_map) > 0:
                        frame_flash = flash_map.get(name, [])
                        merged_coords.extend(frame_flash)

                    if len(merged_coords) > 0:
                        coords_arr = np.unique(np.array(merged_coords, dtype=np.int32), axis=0)
                        xs = coords_arr[:, 0]
                        ys = coords_arr[:, 1]
                        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                        if np.any(valid):
                            xs = xs[valid]
                            ys = ys[valid]
                            gt_vals = gt_gray[ys, xs].astype(np.float64)
                            out_vals = out_gray[ys, xs].astype(np.float64)
                            err = out_vals - gt_vals
                            abs_err = np.abs(err)
                            sq_err = err ** 2

                            blind_abs_sum += float(abs_err.sum())
                            blind_sq_sum += float(sq_err.sum())
                            blind_pix_sum += len(err)

                            in_vals = in_gray[ys, xs].astype(np.float64)
                            in_err = in_vals - gt_vals
                            in_abs = np.abs(in_err)
                            in_sq = in_err ** 2
                            blind_abs_in_sum += float(in_abs.sum())
                            blind_sq_in_sum += float(in_sq.sum())

                            row.update({
                                'blind_mae': float(abs_err.mean()),
                                'blind_rmse': float(np.sqrt(sq_err.mean())),
                                'blind_psnr': 10.0 * np.log10(255.0*255.0 / max(sq_err.mean(), 1e-12)),
                                'blind_mae_input': float(in_abs.mean()),
                                'blind_count': int(len(err))
                            })
                            if row['blind_mae_input'] is not None:
                                row['blind_mae_gain_abs'] = row['blind_mae_input'] - row['blind_mae']
                                row['blind_mae_gain_pct'] = 100.0 * row['blind_mae_gain_abs'] / (row['blind_mae_input'] + 1e-12)

                    per_image_logs.append(row)
                    group_rows.append(row)
                else:
                    # 没有 GT，只保存输出
                    out_img_bgr = cv2.cvtColor(out_img_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(group_pure_dir, name), out_img_bgr)

                if (idx + 1) % 10 == 0:
                    print(f'Processed {idx+1}/{len(group_files)} in group {group_name}')

            # 保存该组的 CSV
            if len(group_rows) > 0:
                group_csv_dir = os.path.join(save_blind_dir, group_name)
                os.makedirs(group_csv_dir, exist_ok=True)
                group_csv = os.path.join(group_csv_dir, 'test_blind_metrics.csv')
                keys = ['image', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr',
                        'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count']
                with open(group_csv, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(group_rows)
                print(f'Per-image test metrics saved to: {group_csv}')

    # 输出全图平均指标
    print('\n===> Full Image Metrics')
    report.print_final_result()

    # 保存全局 CSV
    global_csv = os.path.join(save_blind_dir, 'test_blind_metrics.csv')
    if len(per_image_logs) > 0:
        keys = ['image', 'psnr', 'ssim', 'blind_mae', 'blind_rmse', 'blind_psnr',
                'blind_mae_input', 'blind_mae_gain_abs', 'blind_mae_gain_pct', 'blind_count']
        with open(global_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_image_logs)
        print(f'Per-image test metrics saved to: {global_csv}')

    # 盲元总体统计
    if blind_pix_sum > 0:
        blind_mae = blind_abs_sum / blind_pix_sum
        blind_mse = blind_sq_sum / blind_pix_sum
        blind_rmse = np.sqrt(blind_mse)
        blind_psnr = 10.0 * np.log10(255.0*255.0 / max(blind_mse, 1e-12))
        print('\n===> Blind-Pixel Focused Metrics')
        print(f'BlindCount(total sampled): {blind_pix_sum}')
        print(f'Blind MAE: {blind_mae:.6f} | Blind RMSE: {blind_rmse:.6f} | Blind PSNR: {blind_psnr:.3f}')
        if blind_abs_in_sum > 0:
            blind_mae_in = blind_abs_in_sum / blind_pix_sum
            blind_mse_in = blind_sq_in_sum / blind_pix_sum
            blind_psnr_in = 10.0 * np.log10(255.0*255.0 / max(blind_mse_in, 1e-12))
            gain_abs = blind_mae_in - blind_mae
            gain_pct = 100.0 * gain_abs / (blind_mae_in + 1e-12)
            print(f'Input Blind MAE: {blind_mae_in:.6f} | Input Blind PSNR: {blind_psnr_in:.3f} | MAE Gain: {gain_abs:.6f} ({gain_pct:.2f}%)')
        if len(per_image_logs) > 0:
            print(f'Blind per-image metrics saved to: {global_csv}')
    else:
        print('\nWARN: No blind pixels found in any image. Check your blind coordinate CSV files.')


if __name__ == '__main__':
    main()
