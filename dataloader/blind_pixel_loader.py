"""
盲元补完数据集加载器
支持训练、验证和测试数据集的加载
"""

import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
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
        
        # 数据增强设置
        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=self.flip_prob),
            transforms.RandomRotation(self.rotation_angle),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) if self.enable_augmentation else None
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 预加载根目录兜底的盲元坐标/闪点坐标。
        # 实际优先读取每个 group 目录下的 mask 文件。
        self.root_blind_coords = self._load_coords_csv(self._resolve_optional_path(
            [
                os.path.join(self.mask_dir, config.static_coords_file),
                os.path.join(self.data_root, config.static_coords_file),
                config.static_coords_file,
            ]
        ))
        self.root_flash_map = self._load_flash_csv(self._resolve_optional_path(
            [
                os.path.join(self.mask_dir, config.dynamic_coords_file),
                os.path.join(self.data_root, config.dynamic_coords_file),
                config.dynamic_coords_file,
            ]
        ))
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
        print("  Mask source: group-level blind_pixel_mask.png + blind_pixel_coords.csv + flash_pixel_coords.csv")
    
    def _collect_images(self):
        """收集所有图像，按子文件夹（001, 002等）分组"""
        img_groups = {}
        
        # 获取所有子文件夹
        subdirs = sorted([d for d in os.listdir(self.blur_dir) 
                         if os.path.isdir(os.path.join(self.blur_dir, d))])
        
        for subdir in subdirs:
            blur_subdir = os.path.join(self.blur_dir, subdir)
            sharp_subdir = os.path.join(self.sharp_dir, subdir)
            
            # 收集该子文件夹下的所有PNG文件
            if os.path.isdir(blur_subdir):
                img_names = sorted([f for f in os.listdir(blur_subdir) if f.lower().endswith('.png')], key=natural_sort_key)
                
                img_list = []
                for img_name in img_names:
                    blur_path = os.path.join(blur_subdir, img_name)
                    sharp_path = os.path.join(sharp_subdir, img_name)
                    
                    # 验证文件存在
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
                if reader.fieldnames is None or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
                    return None
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
                if reader.fieldnames is None or 'frame_name' not in reader.fieldnames or 'x' not in reader.fieldnames or 'y' not in reader.fieldnames:
                    return {}
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

    def _create_mask_from_coords(self, coords, shape):
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

    def _load_group_mask_sources(self, group_name):
        group_mask_dir = os.path.join(self.mask_dir, group_name)
        static_mask_path = None
        static_coords_path = None
        flash_coords_path = None

        if os.path.isdir(group_mask_dir):
            for candidate in [
                os.path.join(group_mask_dir, 'blind_pixel_mask.png'),
                os.path.join(group_mask_dir, 'mask.png'),
            ]:
                if os.path.exists(candidate):
                    static_mask_path = candidate
                    break

            for candidate in [
                os.path.join(group_mask_dir, 'blind_pixel_coords.csv'),
                os.path.join(group_mask_dir, 'blind_coords.csv'),
            ]:
                if os.path.exists(candidate):
                    static_coords_path = candidate
                    break

            for candidate in [
                os.path.join(group_mask_dir, 'flash_pixel_coords.csv'),
                os.path.join(group_mask_dir, 'flash_coords.csv'),
            ]:
                if os.path.exists(candidate):
                    flash_coords_path = candidate
                    break

        return static_mask_path, static_coords_path, flash_coords_path

    def _load_group_mask(self, group_name, frame_name, shape):
        cache_key = (group_name, frame_name, shape)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        static_mask_path, static_coords_path, flash_coords_path = self._load_group_mask_sources(group_name)

        mask = None
        if static_mask_path and os.path.exists(static_mask_path):
            mask_img = cv2.imread(static_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if mask_img.shape != shape:
                    mask_img = cv2.resize(mask_img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                # 盲元图中白色(255)表示盲元；模型需要的 mask 是有效区域=1，因此这里取反。
                mask = (mask_img <= 127).astype(np.float32)

        if mask is None and static_coords_path and os.path.exists(static_coords_path):
            coords = self._load_coords_csv(static_coords_path)
            if coords is not None:
                mask = self._create_mask_from_coords(coords, shape)

        if mask is None and self.root_blind_coords is not None:
            mask = self._create_mask_from_coords(self.root_blind_coords, shape)

        if mask is None:
            raise RuntimeError(
                f"Cannot find a valid mask source for group '{group_name}'. "
                f"Expected {os.path.join(self.mask_dir, group_name, 'blind_pixel_mask.png')} "
                f"or a blind coords CSV."
            )

        flash_map = {}
        if flash_coords_path and os.path.exists(flash_coords_path):
            flash_map = self._load_flash_csv(flash_coords_path)
        elif self.root_flash_map:
            flash_map = self.root_flash_map

        flash_coords = flash_map.get(frame_name, [])
        if len(flash_coords) > 0:
            flash_coords = np.array(flash_coords, dtype=np.int32)
            flash_mask = self._create_mask_from_coords(flash_coords, shape)
            mask = np.minimum(mask, flash_mask)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
        self._mask_cache[cache_key] = mask_tensor
        return mask_tensor
    
    def __len__(self):
        """返回数据集大小"""
        return self.total_images
    
    def __getitem__(self, index):
        """获取单个样本"""
        # 找到对应的图像
        current_count = 0
        for group_name, img_list in self.img_groups.items():
            if current_count + len(img_list) > index:
                local_index = index - current_count
                img_info = img_list[local_index]
                break
            current_count += len(img_list)
        
        # 读取图像
        blur_img = cv2.imread(img_info['blur'])
        sharp_img = cv2.imread(img_info['sharp'])
        
        if blur_img is None or sharp_img is None:
            # 如果读取失败，返回零张量
            return self._get_empty_sample()
        
        # BGR -> RGB
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)
        
        # 确保尺寸正确
        if blur_img.shape != (self.image_height, self.image_width, 3):
            blur_img = cv2.resize(blur_img, (self.image_width, self.image_height))
        if sharp_img.shape != (self.image_height, self.image_width, 3):
            sharp_img = cv2.resize(sharp_img, (self.image_width, self.image_height))
        
        # 转换为PIL Image用于transforms
        blur_pil = Image.fromarray(blur_img)
        sharp_pil = Image.fromarray(sharp_img)
        
        # 应用transforms
        if self.enable_augmentation and self.transform_train is not None:
            # 需要相同的随机种子应用相同的增强
            seed = random.randint(0, 2**32 - 1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            blur_tensor = self.transform_train(blur_pil)
            
            random.seed(seed)
            torch.manual_seed(seed)
            sharp_tensor = self.transform_train(sharp_pil)
        else:
            blur_tensor = self.transform_test(blur_pil)
            sharp_tensor = self.transform_test(sharp_pil)
        
        # 按 group 目录读取 mask，并叠加 frame 对应的 flash 像素记录
        mask_tensor = self._load_group_mask(img_info['group'], img_info['name'], (self.image_height, self.image_width))
        
        # 返回字典
        return {
            'blur': blur_tensor,  # [3, H, W]
            'sharp': sharp_tensor,  # [3, H, W]
            'mask': mask_tensor,  # [1, H, W] 或 [3, H, W]
            'img_path': img_info['blur'],
            'group': img_info['group'],
            'name': img_info['name']
        }
    
    def _get_empty_sample(self):
        """返回空样本（当图像加载失败时）"""
        return {
            'blur': torch.zeros((3, self.image_height, self.image_width)),
            'sharp': torch.zeros((3, self.image_height, self.image_width)),
            'mask': torch.ones((1, self.image_height, self.image_width)),
            'img_path': '',
            'group': '',
            'name': ''
        }


def create_dataloader(config, phase='train', shuffle=None, sampler=None):
    """
    创建数据加载器
    
    Args:
        config: 配置对象
        phase: 'train', 'val', 或 'test'
        shuffle: 是否洗牌（默认：训练集为True，其他为False）
    
    Returns:
        DataLoader: PyTorch数据加载器
    """
    if shuffle is None:
        shuffle = (phase == 'train')
    
    dataset = BlindPixelDataset(config, phase=phase)
    
    # 根据phase选择batch size和num_workers
    if phase == 'train':
        batch_size = config.batch_size
        num_workers = config.num_workers
    else:
        batch_size = config.test_batch_size if hasattr(config, 'test_batch_size') else config.batch_size
        num_workers = config.test_num_workers if hasattr(config, 'test_num_workers') else config.num_workers
        shuffle = False  # 测试集不需要洗牌
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(phase == 'train')  # 训练时丢弃不完整的batch
    )
    
    return dataloader
