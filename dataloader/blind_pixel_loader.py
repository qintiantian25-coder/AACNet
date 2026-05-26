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
        
        # 收集所有数据（按子文件夹组织）
        self.img_groups = self._collect_images()
        self.total_images = sum(len(imgs) for imgs in self.img_groups.values())
        
        print(f"Loaded {self.phase} dataset:")
        print(f"  Total groups: {len(self.img_groups)}")
        print(f"  Total images: {self.total_images}")
        print(f"  Blur dir: {self.blur_dir}")
        print(f"  Sharp dir: {self.sharp_dir}")
        print(f"  Mask dir: {self.mask_dir}")
    
    def _collect_images(self):
        """收集所有图像，按子文件夹（001, 002等）分组"""
        img_groups = {}
        
        # 获取所有子文件夹
        subdirs = sorted([d for d in os.listdir(self.blur_dir) 
                         if os.path.isdir(os.path.join(self.blur_dir, d))])
        
        for subdir in subdirs:
            blur_subdir = os.path.join(self.blur_dir, subdir)
            sharp_subdir = os.path.join(self.sharp_dir, subdir)
            mask_subdir = os.path.join(self.mask_dir, subdir)
            
            # 收集该子文件夹下的所有PNG文件
            if os.path.isdir(blur_subdir):
                img_names = sorted([f for f in os.listdir(blur_subdir) if f.endswith('.png')])
                
                img_list = []
                for img_name in img_names:
                    blur_path = os.path.join(blur_subdir, img_name)
                    sharp_path = os.path.join(sharp_subdir, img_name)
                    
                    # 验证文件存在
                    if os.path.exists(blur_path) and os.path.exists(sharp_path):
                        mask_path = os.path.join(mask_subdir, img_name)
                        img_list.append({
                            'blur': blur_path,
                            'sharp': sharp_path,
                            'mask': mask_path if os.path.exists(mask_path) else None,
                            'group': subdir,
                            'name': img_name
                        })
                
                if img_list:
                    img_groups[subdir] = img_list
        
        return img_groups
    
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
        
        # 加载或创建mask
        mask_tensor = self._load_mask(img_info['mask'], (self.image_height, self.image_width))
        
        # 返回字典
        return {
            'blur': blur_tensor,  # [3, H, W]
            'sharp': sharp_tensor,  # [3, H, W]
            'mask': mask_tensor,  # [1, H, W] 或 [3, H, W]
            'img_path': img_info['blur'],
            'group': img_info['group'],
            'name': img_info['name']
        }
    
    def _load_mask(self, mask_path, shape):
        """
        加载mask
        
        Args:
            mask_path: mask文件路径
            shape: 期望的输出形状 (H, W)
        
        Returns:
            mask_tensor: [1, H, W]，1表示有效区域，0表示盲元
        """
        if mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # 确保尺寸正确
                if mask_img.shape != shape:
                    mask_img = cv2.resize(mask_img, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # 二值化：假设 0 = 盲元，>127 = 有效
                mask = (mask_img > 127).astype(np.float32)
                mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
                return mask_tensor
        
        # 如果没有mask文件，返回全1的mask（表示全部有效）
        mask_tensor = torch.ones((1, shape[0], shape[1]), dtype=torch.float32)
        return mask_tensor
    
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
