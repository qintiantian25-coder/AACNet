"""
单张/单目录图像推理脚本 (无 GT / 无 mask 也可运行).

用法:
    # 单张图片
    python inference.py --input /root/Qtt/AACNet/LWIR_test/test_blur/001/frame_001.png

    # 整个目录
    python inference.py --input /root/Qtt/AACNet/LWIR_test/test_blur/001/

    # 指定输出目录
    python inference.py --input ./test_blur/001/ --output ./results_infer

    # 带 blind pixel mask CSV (精确修复)
    python inference.py --input ./test_blur/001/ --mask ./test_mask/001/blind_pixel_coords.csv
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.aacnet import define_g
from util.checkpoint_manager import CheckpointManager

DEFAULTS = {
    'checkpoint': r"/root/Qtt/AACNet/experiments_real/models/best_model.pt",
    'output': r"/root/Qtt/AACNet/results_infer",
    'gpu': 0,
}


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def load_mask_coords(csv_path, h, w):
    """从 CSV 加载盲元坐标，生成 mask (1=valid, 0=blind)"""
    mask = np.ones((h, w), dtype=np.float32)
    if not csv_path or not os.path.exists(csv_path):
        return mask
    coords = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        import csv
        for row in csv.DictReader(f):
            try:
                x, y = int(float(row['x'])), int(float(row['y']))
                if 0 <= x < w and 0 <= y < h:
                    coords.append((x, y))
            except Exception:
                continue
    for x, y in coords:
        mask[y, x] = 0.0
    return mask


def main():
    parser = argparse.ArgumentParser(description='AACNet 推理')
    parser.add_argument('--input', required=True, help='输入图像路径或目录')
    for k, v in DEFAULTS.items():
        parser.add_argument(f'--{k}', default=v)
    parser.add_argument('--mask', default='', help='盲元坐标 CSV (可选)')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    net = define_g(init_type='normal', gpu_ids=[args.gpu] if torch.cuda.is_available() else [],
                   image_size=(512, 640))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('best_model_state_dict', checkpoint.get('last_model_state_dict', checkpoint.get('model_state_dict')))
    if any(k.startswith('module.') for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    print("模型加载完成")

    # 收集输入图片
    if os.path.isfile(args.input):
        img_paths = [args.input]
    else:
        img_paths = sorted([
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if f.lower().endswith(('.png', '.jpg', '.bmp', '.tif', '.tiff'))
        ])

    if not img_paths:
        print("未找到输入图片")
        return

    os.makedirs(args.output, exist_ok=True)
    print(f"处理 {len(img_paths)} 张图片 -> {args.output}")

    with torch.no_grad():
        for path in img_paths:
            name = os.path.splitext(os.path.basename(path))[0]
            img = load_image(path)
            h, w = img.shape

            # 裁剪/填充到 8 的倍数
            pad_h = (8 - h % 8) % 8
            pad_w = (8 - w % 8) % 8
            img_pad = np.pad(img.astype(np.float32), ((0, pad_h), (0, pad_w)), mode='reflect')

            # mask: 1=valid, 0=blind
            mask = load_mask_coords(args.mask, img_pad.shape[0], img_pad.shape[1])

            # 归一化到 [-1, 1]
            img_norm = (img_pad / 255.0) * 2.0 - 1.0

            # 构造 4 通道输入: [blur(RGB), mask]
            blur_3ch = np.stack([img_norm] * 3, axis=0)  # [3, H, W]
            mask_1ch = mask[np.newaxis, ...]               # [1, H, W]
            x = np.concatenate([blur_3ch, mask_1ch], axis=0)  # [4, H, W]
            x = torch.from_numpy(x).unsqueeze(0).float().to(device)

            # 推理
            out, _ = net(x, mask_1ch=torch.from_numpy(mask_1ch).unsqueeze(0).to(device))

            # 后处理
            out = out.squeeze(0).cpu().numpy()  # [3, H, W]
            out = out.transpose(1, 2, 0)        # [H, W, 3]
            out = (out + 1.0) / 2.0 * 255.0
            out = out.clip(0, 255).astype(np.uint8)

            # 裁剪回原始尺寸
            if pad_h > 0 or pad_w > 0:
                out = out[:h, :w, :]

            # 保存 (灰度转回单通道)
            out_gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            save_path = os.path.join(args.output, f'{name}_restored.png')
            cv2.imwrite(save_path, out_gray)
            print(f"  {name}.png -> {save_path}")

    print(f"完成，输出: {args.output}")


if __name__ == '__main__':
    main()
