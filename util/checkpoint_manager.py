"""
检查点管理器 - 负责模型检查点的保存和加载
"""

import os
import torch
import glob
from torch.nn.parallel import DistributedDataParallel as DDP


class CheckpointManager:
    """管理模型检查点"""
    
    def __init__(self, config, metric_name='psnr'):
        """
        初始化检查点管理器
        
        Args:
            config: 配置对象
            metric_name: 用于评估的指标名称 ('psnr' 或 'loss')
        """
        self.config = config
        self.checkpoint_dir = config.checkpoint_dir
        self.model_prefix = config.model_prefix
        self.metric_name = metric_name
        self.best_metric_value = None
        self.best_epoch = None
        self.best_model_state_dict = None
        self.best_optimizer_state_dict = None
        self.best_scheduler_state_dict = None
        self.best_metrics = None
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._cleanup_legacy_checkpoints()

        existing_best = self.find_best_checkpoint()
        if existing_best and os.path.exists(existing_best):
            try:
                checkpoint = torch.load(existing_best, map_location='cpu')
                self.best_metric_value = checkpoint.get('best_metric_value', checkpoint.get('metrics', {}).get(self.metric_name))
                self.best_epoch = checkpoint.get('best_epoch', checkpoint.get('epoch'))
                self.best_model_state_dict = checkpoint.get('best_model_state_dict', checkpoint.get('last_model_state_dict', checkpoint.get('model_state_dict')))
                self.best_optimizer_state_dict = checkpoint.get('best_optimizer_state_dict', checkpoint.get('last_optimizer_state_dict'))
                self.best_scheduler_state_dict = checkpoint.get('best_scheduler_state_dict', checkpoint.get('last_scheduler_state_dict'))
                self.best_metrics = checkpoint.get('best_metrics', checkpoint.get('metrics'))
            except Exception:
                pass

    def _cleanup_legacy_checkpoints(self):
        """删除旧版遗留检查点，保持目录中只剩一个 .pt 文件。"""
        legacy_patterns = [
            os.path.join(self.checkpoint_dir, f'{self.model_prefix}.pth'),
            os.path.join(self.checkpoint_dir, f'{self.model_prefix}_epoch_*.pth'),
        ]
        for pattern in legacy_patterns:
            for legacy_file in glob.glob(pattern):
                try:
                    os.remove(legacy_file)
                    print(f"删除旧检查点: {legacy_file}")
                except Exception:
                    pass

    @staticmethod
    def _state_dict_for_model(model):
        if hasattr(model, 'net_G'):
            if isinstance(model.net_G, (torch.nn.DataParallel, DDP)):
                return model.net_G.module.state_dict()
            return model.net_G.state_dict()
        if isinstance(model, torch.nn.DataParallel):
            return model.module.state_dict()
        if isinstance(model, DDP):
            return model.module.state_dict()
        if hasattr(model, 'state_dict'):
            return model.state_dict()
        raise TypeError(f'Unsupported model type for checkpoint saving: {type(model)}')
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            metrics: 评估指标字典
            is_best: 是否是最好的模型
        """
        save_path = os.path.join(self.checkpoint_dir, f'{self.model_prefix}.pt')

        checkpoint = {}
        if os.path.exists(save_path):
            try:
                checkpoint = torch.load(save_path, map_location='cpu')
            except Exception:
                checkpoint = {}

        last_model_state_dict = self._state_dict_for_model(model)
        last_optimizer_state_dict = optimizer.state_dict() if optimizer else checkpoint.get('last_optimizer_state_dict')
        last_scheduler_state_dict = scheduler.state_dict() if scheduler else checkpoint.get('last_scheduler_state_dict')

        checkpoint.update({
            'epoch': epoch,
            'last_model_state_dict': last_model_state_dict,
            'last_optimizer_state_dict': last_optimizer_state_dict,
            'last_scheduler_state_dict': last_scheduler_state_dict,
            'last_metrics': metrics,
        })

        metric_value = metrics.get(self.metric_name) if isinstance(metrics, dict) else None
        if is_best or (self.best_model_state_dict is None and metric_value is not None):
            self.best_metric_value = metric_value
            self.best_epoch = epoch
            self.best_model_state_dict = last_model_state_dict
            self.best_optimizer_state_dict = last_optimizer_state_dict
            self.best_scheduler_state_dict = last_scheduler_state_dict
            self.best_metrics = metrics

        checkpoint.update({
            'best_epoch': self.best_epoch,
            'best_metric_name': self.metric_name,
            'best_metric_value': self.best_metric_value,
            'best_model_state_dict': self.best_model_state_dict,
            'best_optimizer_state_dict': self.best_optimizer_state_dict,
            'best_scheduler_state_dict': self.best_scheduler_state_dict,
            'best_metrics': self.best_metrics,
        })

        torch.save(checkpoint, save_path)

        if is_best:
            print(f"✓ 当前验证更新了最佳模型: {save_path}")
            print(f"  {self.metric_name}: {self.best_metric_value:.4f}")
        else:
            print(f"  已更新训练状态到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None, load_weights_only=False, use_best=False):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            load_weights_only: 是否只加载权重（不恢复optimizer和epoch）
        
        Returns:
            epoch: 恢复的epoch
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # 加载模型权重
        if use_best and checkpoint.get('best_model_state_dict') is not None:
            model_state = checkpoint['best_model_state_dict']
        else:
            model_state = checkpoint.get('last_model_state_dict', checkpoint.get('model_state_dict'))
        if model_state is None:
            raise KeyError(f"检查点缺少模型权重: {checkpoint_path}")
        
        target_net = None
        if hasattr(model, 'net_G'):
            target_net = model.net_G
        elif isinstance(model, torch.nn.DataParallel):
            target_net = model.module
        elif isinstance(model, DDP):
            target_net = model.module
        elif hasattr(model, 'load_state_dict'):
            target_net = model

        if target_net is None:
            raise TypeError(f'Unsupported model type for checkpoint loading: {type(model)}')

        # 处理DataParallel前缀
        if isinstance(target_net, (torch.nn.DataParallel, DDP)):
            if not any(k.startswith('module.') for k in model_state.keys()):
                model_state = {'module.' + k: v for k, v in model_state.items()}
            target_net.load_state_dict(model_state)
        else:
            if any(k.startswith('module.') for k in model_state.keys()):
                model_state = {k[7:]: v for k, v in model_state.items()}
            target_net.load_state_dict(model_state)
        
        print(f"✓ 加载模型权重")
        
        # 恢复optimizer和scheduler
        if not load_weights_only:
            if optimizer:
                optimizer_state = checkpoint.get('last_optimizer_state_dict', checkpoint.get('optimizer_state_dict'))
                if optimizer_state:
                    optimizer.load_state_dict(optimizer_state)
                print(f"✓ 恢复优化器状态")
            
            if scheduler:
                scheduler_state = checkpoint.get('last_scheduler_state_dict', checkpoint.get('scheduler_state_dict'))
                if scheduler_state:
                    scheduler.load_state_dict(scheduler_state)
                print(f"✓ 恢复学习率调度器状态")
        
        epoch = checkpoint.get('epoch', 0)
        print(f"✓ 恢复到 epoch {epoch}")
        
        return epoch
    
    def find_best_checkpoint(self):
        """找到最好的检查点"""
        best_ckpt = os.path.join(self.checkpoint_dir, f'{self.model_prefix}.pt')
        if os.path.exists(best_ckpt):
            return best_ckpt
        return None
    
    def find_latest_checkpoint(self):
        """找到最新的检查点"""
        pattern = os.path.join(self.checkpoint_dir, f'{self.model_prefix}*.pt')
        ckpt_files = glob.glob(pattern)
        
        if not ckpt_files:
            return None
        
        # 按修改时间排序，返回最新的
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        return latest_ckpt
    
    def remove_old_checkpoints(self, keep_num=3):
        """
        删除旧的检查点（只保留最新的keep_num个）
        
        Args:
            keep_num: 保留的检查点数量
        """
        pattern = os.path.join(self.checkpoint_dir, f'{self.model_prefix}*.pt')
        ckpt_files = sorted(glob.glob(pattern), key=os.path.getmtime)
        
        # 最好的模型总是保留
        best_ckpt = os.path.join(self.checkpoint_dir, f'{self.model_prefix}.pt')
        
        to_remove = []
        for ckpt in ckpt_files[:-keep_num]:
            if ckpt != best_ckpt:
                to_remove.append(ckpt)
        
        for ckpt in to_remove:
            try:
                os.remove(ckpt)
                print(f"删除旧的检查点: {ckpt}")
            except Exception as e:
                print(f"删除检查点失败: {ckpt}, 错误: {e}")
