"""
检查点管理器 - 负责模型检查点的保存和加载
"""

import os
import torch
import glob


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
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
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
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict() if not isinstance(model, torch.nn.DataParallel) else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }
        
        if is_best:
            # 保存最好的模型（覆盖之前的）
            save_path = os.path.join(self.checkpoint_dir, f'{self.model_prefix}.pth')
            torch.save(checkpoint, save_path)
            
            # 更新最好的指标值
            self.best_metric_value = metrics.get(self.metric_name)
            
            print(f"✓ 保存最佳模型到: {save_path}")
            print(f"  {self.metric_name}: {self.best_metric_value:.4f}")
        
        # 如果配置了保存间隔，也保存临时检查点
        if self.config.save_interval > 0 and epoch % self.config.save_interval == 0:
            temp_save_path = os.path.join(
                self.checkpoint_dir,
                f'{self.model_prefix}_epoch_{epoch}.pth'
            )
            torch.save(checkpoint, temp_save_path)
            print(f"  临时检查点: {temp_save_path}")
    
    def load_checkpoint(self, checkpoint_path, model, optimizer=None, scheduler=None, load_weights_only=False):
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
        model_state = checkpoint['model_state_dict']
        
        # 处理DataParallel前缀
        if isinstance(model, torch.nn.DataParallel):
            # 如果模型是DataParallel但检查点没有'module.'前缀
            if not any(k.startswith('module.') for k in model_state.keys()):
                model_state = {'module.' + k: v for k, v in model_state.items()}
            model.load_state_dict(model_state)
        else:
            # 如果模型不是DataParallel但检查点有'module.'前缀
            if any(k.startswith('module.') for k in model_state.keys()):
                model_state = {k[7:]: v for k, v in model_state.items()}
            model.load_state_dict(model_state)
        
        print(f"✓ 加载模型权重")
        
        # 恢复optimizer和scheduler
        if not load_weights_only:
            if optimizer and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict']:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print(f"✓ 恢复优化器状态")
            
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"✓ 恢复学习率调度器状态")
        
        epoch = checkpoint.get('epoch', 0)
        print(f"✓ 恢复到 epoch {epoch}")
        
        return epoch
    
    def find_best_checkpoint(self):
        """找到最好的检查点"""
        best_ckpt = os.path.join(self.checkpoint_dir, f'{self.model_prefix}.pth')
        if os.path.exists(best_ckpt):
            return best_ckpt
        return None
    
    def find_latest_checkpoint(self):
        """找到最新的检查点"""
        pattern = os.path.join(self.checkpoint_dir, f'{self.model_prefix}*.pth')
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
        pattern = os.path.join(self.checkpoint_dir, f'{self.model_prefix}*.pth')
        ckpt_files = sorted(glob.glob(pattern), key=os.path.getmtime)
        
        # 最好的模型总是保留
        best_ckpt = os.path.join(self.checkpoint_dir, f'{self.model_prefix}.pth')
        
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
