#!/usr/bin/env python3
"""验证组级 mask 文件的存在性与读取语义（用于 val/test mask 目录验证）。

用法示例：
  python scripts/verify_mask_groups.py --mask_root /path/to/val_mask --sample_image_h 512 --sample_image_w 512

脚本会遍历 mask_root 下的子目录（如 001, 002），对每个 group：
 - 判断是否存在 blind_pixel_mask.png, blind_pixel_coords.csv, flash_pixel_coords.csv
 - 若存在 mask png，读取为灰度并展示原始统计与反转之后的统计（用于确认 255 表示盲元，被转换为 0）
 - 若存在 coords csv，读取并报告坐标点数
"""

import os
import argparse
import csv
import numpy as np
import cv2


def load_coords_csv(csv_path):
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


def inspect_mask_png(png_path, sample_shape=None):
    img = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    orig_shape = img.shape
    if sample_shape and orig_shape != sample_shape:
        img = cv2.resize(img, (sample_shape[1], sample_shape[0]), interpolation=cv2.INTER_NEAREST)
    # 在数据集中白色(255)代表盲元——模型期望有效区域=1，所以取反：mask = (img <= 127)
    inverted = (img <= 127).astype(np.uint8)
    stats = {
        'orig_shape': orig_shape,
        'min_orig': int(img.min()),
        'max_orig': int(img.max()),
        'unique_orig': int(len(np.unique(img))),
        'sum_orig': int(int(img.sum())),
        'inverted_sum': int(int(inverted.sum())),
        'inverted_unique': int(len(np.unique(inverted)))
    }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_root', required=True, help='val/test mask 根目录，包含 001/002/... 子文件夹')
    parser.add_argument('--sample_image_h', type=int, default=None, help='可选的采样高度，用于 resize mask 便于比较')
    parser.add_argument('--sample_image_w', type=int, default=None, help='可选的采样宽度')
    args = parser.parse_args()

    mask_root = args.mask_root
    if not os.path.isdir(mask_root):
        print(f'ERROR: mask_root 不存在或不是目录: {mask_root}')
        return

    subdirs = sorted([d for d in os.listdir(mask_root) if os.path.isdir(os.path.join(mask_root, d))])
    if not subdirs:
        print(f'WARN: 在 {mask_root} 下未发现任何子目录')
        return

    sample_shape = None
    if args.sample_image_h and args.sample_image_w:
        sample_shape = (args.sample_image_h, args.sample_image_w)

    for group in subdirs:
        group_dir = os.path.join(mask_root, group)
        print(f'-- Group: {group} --')
        png_candidates = [
            os.path.join(group_dir, 'blind_pixel_mask.png'),
            os.path.join(group_dir, 'blind_pixel_mask.jpg'),
            os.path.join(group_dir, 'mask.png'),
        ]
        coords_candidates = [
            os.path.join(group_dir, 'blind_pixel_coords.csv'),
            os.path.join(group_dir, 'blind_coords.csv'),
        ]
        flash_candidates = [
            os.path.join(group_dir, 'flash_pixel_coords.csv'),
            os.path.join(group_dir, 'flash_coords.csv'),
        ]

        png_found = None
        for p in png_candidates:
            if os.path.exists(p):
                png_found = p
                break

        coords_found = None
        for c in coords_candidates:
            if os.path.exists(c):
                coords_found = c
                break

        flash_found = None
        for f in flash_candidates:
            if os.path.exists(f):
                flash_found = f
                break

        print(' mask png:', png_found if png_found else 'NOT FOUND')
        print(' coords csv:', coords_found if coords_found else 'NOT FOUND')
        print(' flash csv:', flash_found if flash_found else 'NOT FOUND')

        if png_found:
            stats = inspect_mask_png(png_found, sample_shape)
            if stats is None:
                print('  Failed to read png')
            else:
                print(f"  orig_shape={stats['orig_shape']} min={stats['min_orig']} max={stats['max_orig']} unique={stats['unique_orig']} sum={stats['sum_orig']}")
                print(f"  inverted_sum={stats['inverted_sum']} inverted_unique={stats['inverted_unique']}")

        if coords_found:
            coords = load_coords_csv(coords_found)
            if coords is None:
                print('  coords csv parsed but no x/y fields or empty')
            else:
                print(f'  coords count: {len(coords)}')

        print('')


if __name__ == '__main__':
    main()
