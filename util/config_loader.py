"""
配置文件加载器 - 从experiment.cfg读取配置
"""

import configparser
import os
from argparse import Namespace
import json


class ConfigLoader:
    """从.cfg文件加载配置的工具类"""
    
    def __init__(self, config_path):
        """
        初始化配置加载器
        
        Args:
            config_path (str): 配置文件路径
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        self.config_path = config_path
        self.parser = configparser.ConfigParser()
        self.parser.read(config_path, encoding='utf-8')
    
    def get_config(self):
        """
        获取所有配置作为Namespace对象
        
        Returns:
            argparse.Namespace: 包含所有配置的命名空间对象
        """
        config = Namespace()
        
        # 数据集配置
        config.data_root = self.get('dataset', 'data_root', './data')
        config.train_blur_dir = self.get('dataset', 'train_blur_dir', 'train_blur')
        config.train_sharp_dir = self.get('dataset', 'train_sharp_dir', 'train_sharp')
        config.train_mask_dir = self.get('dataset', 'train_mask_dir', 'train_mask')
        config.val_blur_dir = self.get('dataset', 'val_blur_dir', 'val_blur')
        config.val_sharp_dir = self.get('dataset', 'val_sharp_dir', 'val_sharp')
        config.val_mask_dir = self.get('dataset', 'val_mask_dir', 'val_mask')
        config.test_blur_dir = self.get('dataset', 'test_blur_dir', 'test_blur')
        config.test_sharp_dir = self.get('dataset', 'test_sharp_dir', 'test_sharp')
        config.test_mask_dir = self.get('dataset', 'test_mask_dir', 'test_mask')
        
        # 图像尺寸
        config.image_width = self.getint('dataset', 'image_width', 640)
        config.image_height = self.getint('dataset', 'image_height', 512)
        
        # 数据增强
        config.enable_augmentation = self.getbool('dataset', 'enable_augmentation', True)
        config.flip_prob = self.getfloat('dataset', 'flip_prob', 0.5)
        config.rotation_angle = self.getint('dataset', 'rotation_angle', 10)
        
        # 训练配置
        config.num_epochs = self.getint('training', 'num_epochs', 100)
        config.batch_size = self.getint('training', 'batch_size', 2)
        config.learning_rate = self.getfloat('training', 'learning_rate', 0.0001)
        config.lr_schedule = self.get('training', 'lr_schedule', 'exponential')
        # 兼容 BaseModel 调度器参数命名：将 lr_schedule 映射为 lr_policy
        lr_policy_alias = {
            'exponential': 'exponent',
            'exp': 'exponent',
            'cos': 'cosine',
        }
        schedule_key = str(config.lr_schedule).strip().lower()
        config.lr_policy = lr_policy_alias.get(schedule_key, schedule_key)

        # 供 get_scheduler 使用的参数（缺失时回退到 num_epochs）
        config.iter_count = self.getint('training', 'iter_count', 1)
        config.niter = self.getint('training', 'niter', config.num_epochs)
        config.niter_decay = self.getint('training', 'niter_decay', 0)
        config.lr_decay_iters = self.getint('training', 'lr_decay_iters', 50)
        config.eta_min = self.getfloat('training', 'eta_min', 0.0)
        config.lr_decay_factor = self.getfloat('training', 'lr_decay_factor', 0.99)
        config.val_interval = self.getint('training', 'val_interval', 20)
        config.num_workers = self.getint('training', 'num_workers', 8)
        config.shuffle = self.getbool('training', 'shuffle', True)
        
        # 优化器配置
        config.optimizer_type = self.get('optimizer', 'optimizer_type', 'adam')
        config.adam_beta1 = self.getfloat('optimizer', 'adam_beta1', 0.5)
        config.adam_beta2 = self.getfloat('optimizer', 'adam_beta2', 0.9)
        config.adam_weight_decay = self.getfloat('optimizer', 'adam_weight_decay', 0)
        config.sgd_momentum = self.getfloat('optimizer', 'sgd_momentum', 0.9)
        
        # 损失函数配置
        config.lambda_l1 = self.getfloat('loss', 'lambda_l1', 1.0)
        config.lambda_perceptual = self.getfloat('loss', 'lambda_perceptual', 1.0)
        config.lambda_style = self.getfloat('loss', 'lambda_style', 250.0)
        config.lambda_adv = self.getfloat('loss', 'lambda_adv', 0.1)
        config.lambda_consist = self.getfloat('loss', 'lambda_consist', 1.0)
        
        # 模型配置
        config.model_name = self.get('model', 'model_name', 'aacnet')
        config.generator_ngf = self.getint('model', 'generator_ngf', 48)
        config.discriminator_ndf = self.getint('model', 'discriminator_ndf', 64)
        config.in_channels = self.getint('model', 'in_channels', 3)
        config.out_channels = self.getint('model', 'out_channels', 3)
        
        # 测试配置
        config.test_batch_size = self.getint('testing', 'batch_size', 1)
        config.test_num_workers = self.getint('testing', 'num_workers', 4)
        config.compute_metrics = self.getbool('testing', 'compute_metrics', True)
        config.crop_border = self.getint('testing', 'crop_border', 0)
        config.save_results = self.getbool('testing', 'save_results', True)
        config.results_dir = self.get('testing', 'results_dir', './results')
        
        # 检查点配置
        config.checkpoint_dir = self.get('checkpoint', 'checkpoint_dir', './checkpoints')
        config.model_prefix = self.get('checkpoint', 'model_prefix', 'best_model')
        config.save_interval = self.getint('checkpoint', 'save_interval', 0)
        config.save_best_only = self.getbool('checkpoint', 'save_best_only', True)
        config.best_metric = self.get('checkpoint', 'best_metric', 'psnr')
        
        # 日志配置
        config.log_dir = self.get('logging', 'log_dir', './logs')
        config.print_freq = self.getint('logging', 'print_freq', 10)
        config.save_val_visual = self.getbool('logging', 'save_val_visual', True)
        config.val_visual_dir = self.get('logging', 'val_visual_dir', './val_results')
        
        # 设备配置
        config.gpu_ids = self.get('device', 'gpu_ids', '0')
        # 将gpu_ids字符串转换为整数列表
        if isinstance(config.gpu_ids, str):
            if config.gpu_ids.lower() == 'cpu' or config.gpu_ids == '':
                config.gpu_ids = []
            else:
                config.gpu_ids = [int(i.strip()) for i in config.gpu_ids.split(',')]
        config.use_dataparallel = self.getbool('device', 'use_dataparallel', False)
        config.mixed_precision = self.getbool('device', 'mixed_precision', False)
        
        # 盲元配置
        config.compute_blind_metrics = self.getbool('blind_pixel', 'compute_blind_metrics', True)
        config.mask_type = self.get('blind_pixel', 'mask_type', 'both')
        config.static_coords_file = self.get('blind_pixel', 'static_coords_file', 'blind_coords.csv')
        config.dynamic_coords_file = self.get('blind_pixel', 'dynamic_coords_file', 'flash_pixel_coords.csv')
        
        # 恢复训练配置
        config.resume_training = self.getbool('resume', 'resume_training', False)
        config.checkpoint_path = self.get('resume', 'checkpoint_path', '')
        config.load_weights_only = self.getbool('resume', 'load_weights_only', False)
        
        # 其他配置
        config.seed = self.getint('misc', 'seed', 42)
        config.deterministic = self.getbool('misc', 'deterministic', False)
        config.cudnn_benchmark = self.getbool('misc', 'cudnn_benchmark', True)
        
        # 派生配置
        config.isTrain = False  # 将在main.py中设置
        
        return config
    
    def get(self, section, option, default=''):
        """获取字符串值"""
        try:
            return self.parser.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def getint(self, section, option, default=0):
        """获取整数值"""
        try:
            return self.parser.getint(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def getfloat(self, section, option, default=0.0):
        """获取浮点值"""
        try:
            return self.parser.getfloat(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def getbool(self, section, option, default=False):
        """获取布尔值"""
        try:
            value = self.parser.get(section, option).lower()
            if value in ('true', 'yes', '1', 'on'):
                return True
            elif value in ('false', 'no', '0', 'off'):
                return False
            else:
                return default
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def print_config(self):
        """打印所有配置信息"""
        print("\n" + "="*60)
        print("配置信息 (Config)")
        print("="*60)
        
        for section in self.parser.sections():
            print(f"\n[{section}]")
            for option in self.parser.options(section):
                value = self.parser.get(section, option)
                print(f"  {option} = {value}")
        
        print("="*60 + "\n")
