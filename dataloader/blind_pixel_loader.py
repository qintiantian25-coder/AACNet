"""
盲元补完数据集加载器
支持训练、验证和测试数据集的加载
已完美适配 fangzhen_adaptive 生成的静态盲元与动态闪元数据，并修正同步数据增强 Bug
"""

import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
from PIL import Image
import random
import csv


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in __import__('re').split('([0-9]+)', s)]


class BlindPixelDataset(data.Dataset):
    """盲元补完数据集"""
    
    def __init__(self, config, phase='train'):
        """
        初始化数据集
        
        Args:
            config: 配置对象（从experiment.cfg加载）
            phase: 'train', 'val', 或 'test'
        """
        self.config = config
        self.phase = phase  # train, val, test
        self.data_root = config.data_root
        
        # 根据phase选择数据目录
        if phase == 'train':
            self.blur_dir = os.path.join(self.data_root, config.train_blur_dir)
            self.sharp_dir = os.path.join(self.data_root, config.train_sharp_dir)
            self.mask_dir = os.path.join(self.data_root, config.train_mask_dir)
        elif phase == 'val':
            self.blur_dir = os.path.join(self.data_root, config.val_blur_dir)
            self.sharp_dir = os.path.join(self.data_root, config.val_sharp_dir)
            self.mask_dir = os.path.join(self.data_root, config.val_mask_dir)
        else:  # test
            self.blur_dir = os.path.join(self.data_root, config.test_blur_dir)
            self.sharp_dir = os.path.join(self.data_root, config.test_sharp_dir)
            self.mask_dir = os.path.join(self.data_root, config.test_mask_dir)
        
        # 图像尺寸
        self.image_width = config.image_width
        self.image_height = config.image_height
        
        # 数据增强参数（仅用于训练）
        self.enable_augmentation = config.enable_augmentation and (phase == 'train')
        self.flip_prob = config.flip_prob
        self.rotation_angle = config.rotation_angle
        
        # 预加载根目录或全局的动态闪点坐标（优先从group目录读取，这里做兜底）
        dynamic_file = config.dynamic_coords_file if hasattr(config, 'dynamic_coords_file') else 'flash_pixel_coords.csv'
        self.root_flash_map = self._load_flash_csv(self._resolve_optional_path(
            [
                os.path.join(self.mask_dir, dynamic_file),
                os.path.join(self.data_root, dynamic_file),
                dynamic_file,
            ]
        ))
        
        # 缓存每个 group 的闪元映射，避免重复解析 CSV
        self._group_flash_maps = {}
        self._mask_cache = {}
        
        # 收集所有数据（按子文件夹组织）
        self.img_groups = self._collect_images()
        self.total_images = sum(len(imgs) for imgs in self.img_groups.values())
        
        print(f"Loaded {self.phase} dataset:")
        print(f"  Total groups: {len(self.img_groups)}")
        print(f"  Total images: {self.total_images}")
        print(f"  Blur dir: {self.blur_dir}")
        print(f"  Sharp dir: {self.sharp_dir}")
        print(f"  Mask dir: {self.mask_dir}")
        print("  [SUCCESS] Updated loader: Mask now changes dynamically with flash pixels per frame!")
        print("  [SUCCESS] Updated loader: Synchronized image & mask data augmentation implemented.")
    
    def _collect_images(self):
        """收集所有图像，按子文件夹（001, 002等）分组"""
        img_groups = {}
        if not os.path.exists(self.blur_dir):
            return img_groups
            
        subdirs = sorted([d for d in os.listdir(self.blur_dir) 
                         if os.path.isdir(os.path.join(self.blur_dir, d))])
        
        for subdir in subdirs:
            blur_subdir = os.path.join(self.blur_dir, subdir)
            sharp_subdir = os.path.join(self.sharp_dir, subdir)
            
            if os.path.isdir(blur_subdir):
                img_names = sorted([f for f in os.listdir(blur_subdir) if f.lower().endswith('.png')], key=natural_sort_key)
                
                img_list = []
                for img_name in img_names:
                    blur_path = os.path.join(blur_subdir, img_name)
                    sharp_path = os.path.join(sharp_subdir, img_name)
                    
                    if os.path.exists(blur_path) and os.path.exists(sharp_path):
                        img_list.append({
                            'blur': blur_path,
                            'sharp': sharp_path,
                            'group': subdir,
                            'name': img_name
                        })
                
                if img_list:
                    img_groups[subdir] = img_list
        
        return img_groups

    def _resolve_optional_path(self, candidates):
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _load_coords_csv(self, csv_path):
        if not csv_path or not os.path.exists(csv_path):
            return None
        coords = []
        try:
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        coords.append((int(float(row['x'])), int(float(row['y']))))
                    except Exception:
                        continue
        except Exception:
            return None
        if len(coords) == 0:
            return None
        return np.unique(np.array(coords, dtype=np.int32), axis=0)

    def _load_flash_csv(self, csv_path):
        if not csv_path or not os.path.exists(csv_path):
            return {}
        flash_map = {}
        try:
            with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        fname = os.path.basename(str(row['frame_name']))
                        x = int(float(row['x']))
                        y = int(float(row['y']))
                    except Exception:
                        continue
                    flash_map.setdefault(fname, set()).add((x, y))
        except Exception:
            return {}
        return {k: list(v) for k, v in flash_map.items()}

    def _get_group_flash_map(self, group_name):
        """获取或缓存特定子组的动态闪点映射"""
        if group_name in self._group_flash_maps:
            return self._group_flash_maps[group_name]
            
        group_mask_dir = os.path.join(self.mask_dir, group_name)
        flash_csv_path = None
        for candidate in ['flash_pixel_coords.csv', 'flash_coords.csv']:
            p = os.path.join(group_mask_dir, candidate)
            if os.path.exists(p):
                flash_csv_path = p
                break
                
        if flash_csv_path:
            g_flash_map = self._load_flash_csv(flash_csv_path)
        else:
            g_flash_map = self.root_flash_map
            
        self._group_flash_maps[group_name] = g_flash_map
        return g_flash_map

    def _create_mask_from_coords(self, coords, shape):
        # 模型要求：1.0表示有效像素，0.0表示需要补完的盲元
        mask = np.ones(shape, dtype=np.float32)
        if coords is None or len(coords) == 0:
            return mask
        h, w = shape
        xs = coords[:, 0]
        ys = coords[:, 1]
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if np.any(valid):
            mask[ys[valid], xs[valid]] = 0.0
        return mask

    def _load_base_static_mask(self, group_name, shape):
        """读取静态盲元基础遮罩（不含当帧闪点）"""
        cache_key = (group_name, shape)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key].copy()

        group_mask_dir = os.path.join(self.mask_dir, group_name)
        static_coords_path = None
        static_mask_path = None

        if os.path.isdir(group_mask_dir):
            for c in ['blind_pixel_coords.csv', 'blind_coords.csv']:
                p = os.path.join(group_mask_dir, c)
                if os.path.exists(p):
                    static_coords_path = p
                    break
            for c in ['blind_pixel_mask.png', 'mask.png']:
                p = os.path.join(group_mask_dir, c)
                if os.path.exists(p):
                    static_mask_path = p
                    break

        mask = None
        # 1. 优先从 csv 坐标重建精密遮罩
        if static_coords_path:
            coords = self._load_coords_csv(static_coords_path)
            if coords is not None:
                mask = self._create_mask_from_coords(coords, shape)

        # 2. 次选从盲元遮罩图像读取
        if mask flee and static_mask_path:
            mask_img = cv2.imread(static_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if mask_img.shape != shape:
                    mask_img = cv2.resize(mask_img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                # fangzhen_adaptive中，盲元渲染像素很低(黑色靠近0)，或者用掩码保存时可能非0。
                # 统一规则：在原始仿真代码中，非盲元处保持原图，盲元处涂黑。所以掩码图像中完好区域>127，盲元<=127
                mask = (mask_img > 127).astype(np.float32)

        # 3. 都没有则用全局静态坐标兜底
        if mask is None and hasattr(self, 'root_blind_coords') and self.root_blind_coords is not None:
            mask = self._create_mask_from_coords(self.root_blind_coords, shape)

        if mask is None:
            # 如果都找不到，默认全1（无静态盲元）
            mask = np.ones(shape, dtype=np.float32)

        self._mask_cache[cache_key] = mask
        return mask.copy()

    def __len__(self):
        return self.total_images
    
    def __getitem__(self, index):
        current_count = 0
        img_info = None
        for group_name, img_list in self.img_groups.items():
            if current_count + len(img_list) > index:
                img_info = img_list[index - current_count]
                break
            current_count += len(img_list)
        
        if img_info is None:
            return self._get_empty_sample()
            
        # 读取图像
        blur_img = cv2.imread(img_info['blur'])
        sharp_img = cv2.imread(img_info['sharp'])
        
        if blur_img is None or sharp_img is None:
            return self._get_empty_sample()
        
        # BGR -> RGB
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        
        # 尺寸规范化
        if blur_img.shape[:2] != (self.image_height, self.image_width):
            blur_img = cv2.resize(blur_img, (self.image_width, self.image_height))
        if sharp_img.shape[:2] != (self.image_height, self.image_width):
            sharp_img = cv2.resize(sharp_img, (self.image_width, self.image_height))
            
        # 1. 动态获取并组装当帧的完整 Mask (静态盲元 + 这一帧特有的动态闪元)
        mask = self._load_base_static_mask(img_info['group'], (self.image_height, self.image_width))
        
        flash_map = self._get_group_flash_map(img_info['group'])
        frame_name = img_info['name']
        if frame_name in flash_map:
            for x, y in flash_map[frame_name]:
                if 0 <= y < self.image_height and 0 <= x < self.image_width:
                    mask[y, x] = 0.0  # 将动态闪点所在位置也在 Mask 中扣除
        
        # 2. 同步执行数据增强 (完全杜绝数据对不齐的问题)
        if self.enable_augmentation:
            # 随机水平翻转
            if random.random() < self.flip_prob:
                blur_img = cv2.flip(blur_img, 1)
                sharp_img = cv2.flip(sharp_img, 1)
                mask = cv2.flip(mask, 1)
                
            # 随机旋转
            angle = random.uniform(-self.rotation_angle, self.rotation_angle)
            if abs(angle) > 1e-2:
                M = cv2.getRotationMatrix2D((self.image_width / 2, self.image_height / 2), angle, 1.0)
                blur_img = cv2.warpAffine(blur_img, M, (self.image_width, self.image_height), flags=cv2.INTER_LINEAR)
                sharp_img = cv2.warpAffine(sharp_img, M, (self.image_width, self.image_height), flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(mask, M, (self.image_width, self.image_height), flags=cv2.INTER_NEAREST, borderValue=1.0)

        # 3. 图像归一化至 [-1, 1]，Mask 转换为 Torch Tensor 保持 [0, 1]
        blur_tensor = torch.from_numpy(blur_img.transpose(2, 0, 1)).float() / 127.5 - 1.0
        sharp_tensor = torch.from_numpy(sharp_img.transpose(2, 0, 1)).float() / 127.5 - 1.0
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()  # 形状为 [1, H, W]
        
        return {
            'blur': blur_tensor,
            'sharp': sharp_tensor,
            'mask': mask_tensor,
            'img_path': img_info['blur'],
            'group': img_info['group'],
            'name': img_info['name']
        }
    
    def _get_empty_sample(self):
        return {
            'blur': torch.zeros((3, self.image_height, self.image_width)),
            'sharp': torch.zeros((3, self.image_height, self.image_width)),
            'mask': torch.ones((1, self.image_height, self.image_width)),
            'img_path': '',
            'group': '',
            'name': ''
        }


def create_dataloader(config, phase='train', shuffle=None, sampler=None):
    if shuffle is None:
        shuffle = (phase == 'train')
    
    dataset = BlindPixelDataset(config, phase=phase)
    
    if phase == 'train':
        batch_size = config.batch_size
        num_workers = config.num_workers
    else:
        batch_size = config.test_batch_size if hasattr(config, 'test_batch_size') else config.batch_size
        num_workers = config.test_num_workers if hasattr(config, 'test_num_workers') else config.num_workers
        shuffle = False
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == 'train')
    )
    
    return dataloader