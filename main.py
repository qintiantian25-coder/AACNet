"""
AACNet 盲元补完网络 - 主训练和测试脚本

用法:
    训练: python main.py --train --config_path ./experiment.cfg
    测试: python main.py --test --config_path ./experiment.cfg
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import csv
import cv2
from datetime import datetime
import shutil
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 导入项目模块
from util.config_loader import ConfigLoader
from dataloader.blind_pixel_loader import BlindPixelDataset, create_dataloader
from model import create_model
from util.metrics import MetricCalculator
from util.checkpoint_manager import CheckpointManager
from util.logger import Logger


def setup_seed(config):
    """设置随机种子以保证可重现性"""
    if config.seed != -1:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
    
    if config.deterministic:
        torch.backends.cudnn.deterministic = True


def setup_device(config):
    """设置计算设备"""
    if torch.cuda.is_available() and len(config.gpu_ids) > 0:
        torch.cuda.set_device(config.gpu_ids[0])
        device = torch.device(f'cuda:{config.gpu_ids[0]}')
        print(f"Using single GPU: {config.gpu_ids[0]}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def setup_directories(config):
    """创建必要的目录"""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    if config.save_val_visual:
        os.makedirs(config.val_visual_dir, exist_ok=True)
    if config.save_results:
        os.makedirs(config.results_dir, exist_ok=True)


def set_model_train_mode(model):
    """Set the underlying network(s) to train mode."""
    if hasattr(model, 'net_G'):
        model.net_G.train()
    elif hasattr(model, 'train'):
        model.train()


def set_model_eval_mode(model):
    """Set the underlying network(s) to eval mode."""
    if hasattr(model, 'net_G'):
        model.net_G.eval()
    elif hasattr(model, 'eval'):
        model.eval()


def init_distributed(config):
    """单卡模式下不初始化分布式环境。"""
    config.distributed = False
    config.local_rank = 0
    config.rank = 0
    config.world_size = 1
    config.is_main_process = True
    return False


def train(config, device):
    """
    训练函数
    
    Args:
        config: 配置对象
        device: 计算设备
    """
    print("\n" + "="*60)
    print("开始训练 (Training)")
    print("="*60)
    
    is_main_process = getattr(config, 'is_main_process', True)
    is_distributed = getattr(config, 'distributed', False)

    # 创建数据加载器
    print("\n正在加载数据...")
    train_sampler = None
    if is_distributed:
        train_dataset = BlindPixelDataset(config, phase='train')
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            drop_last=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = create_dataloader(config, phase='train', shuffle=True)

    val_loader = create_dataloader(config, phase='val', shuffle=False) if is_main_process else None
    
    print(f"  训练集大小: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"  验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    print("\n正在构建模型...")
    config.model = config.model_name
    config.lr = config.learning_rate
    config.isTrain = True
    
    if not hasattr(config, 'checkpoints_dir'):
        config.checkpoints_dir = config.checkpoint_dir
    if not hasattr(config, 'name'):
        config.name = getattr(config, 'experiment_name', 'aacnet_blind')
    if not hasattr(config, 'which_iter'):
        config.which_iter = 0
    if not hasattr(config, 'continue_train'):
        config.continue_train = False
    
    model = create_model(config)

    # 显存优化：在 DDP 包装之前，先将底层的 net_G 移动到当前 rank 设备上
    if hasattr(model, 'net_G'):
        model.net_G = model.net_G.to(device)
    
    if is_distributed and config.world_size > 1 and hasattr(model, 'net_G'):
        model.net_G = DDP(
            model.net_G,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
    print(f"  模型: {config.model_name}")
    
    # 创建检查点管理器
    checkpoint_manager = CheckpointManager(config, config.best_metric)
    
    # 创建日志记录器（仅主进程写日志）
    train_logger = Logger(config, log_name='training') if is_main_process else None
    val_logger = Logger(config, log_name='validation') if is_main_process else None
    if train_logger is not None:
        train_logger.log_config(config)

    if is_distributed:
        dist.barrier()
    
    # 恢复训练（如果启用）
    start_epoch = 0
    if config.resume_training and config.checkpoint_path:
        print(f"\n从训练状态恢复: {config.checkpoint_path}")
        start_epoch = checkpoint_manager.load_checkpoint(
            config.checkpoint_path,
            model,
            load_weights_only=config.load_weights_only,
            use_best=False
        )
    elif config.resume_training:
        latest_ckpt = checkpoint_manager.find_latest_checkpoint()
        if latest_ckpt:
            print(f"\n从最新训练状态恢复: {latest_ckpt}")
            start_epoch = checkpoint_manager.load_checkpoint(
                latest_ckpt,
                model,
                load_weights_only=config.load_weights_only,
                use_best=False
            )
    
    # 训练循环
    print(f"\n开始训练 ({start_epoch}/{config.num_epochs} epochs)")
    print("-" * 60)
    
    best_metric = None
    metric_calc = MetricCalculator(config.crop_border)
    optimizer = getattr(model, 'optimizers', [None])[0] if hasattr(model, 'optimizers') and len(getattr(model, 'optimizers', [])) > 0 else None
    scheduler = getattr(model, 'schedulers', [None])[0] if hasattr(model, 'schedulers') and len(getattr(model, 'schedulers', [])) > 0 else None
    
    for epoch in range(start_epoch, config.num_epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 训练阶段
        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch, train_logger)

        if is_distributed:
            loss_tensor = torch.tensor(train_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = loss_tensor.item() / config.world_size
        
        # 获取当前学习率
        if hasattr(model, 'schedulers') and len(model.schedulers) > 0:
            model.update_learning_rate()
            current_lr = model.optimizers[0].param_groups[0]['lr']
            if train_logger is not None:
                train_logger.log(f"Epoch {epoch + 1}/{config.num_epochs} - LR: {current_lr:.6f}")

        # 保存当前训练状态（统一保存在主进程）
        if is_main_process:
            checkpoint_manager.save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                {'loss': train_loss},
                is_best=False
            )
        
        # 验证阶段
        if (epoch + 1) % config.val_interval == 0:
            if is_distributed:
                dist.barrier()  # 验证前同步：所有人停下，准备跑验证

            val_metrics = validate(model, val_loader, device, config, epoch, train_logger, val_logger, metric_calc) if is_main_process else None
            
            # 检查是否是最好的模型
            if is_main_process:
                current_metric = val_metrics[config.best_metric]
                
                if best_metric is None:
                    best_metric = current_metric
                    is_best = True
                elif config.best_metric == 'psnr':
                    is_best = current_metric > best_metric
                    if is_best:
                        best_metric = current_metric
                else:
                    is_best = current_metric < best_metric
                    if is_best:
                        best_metric = current_metric
                
                # 保存检查点
                if is_best or not config.save_best_only:
                    checkpoint_manager.save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        epoch + 1,
                        val_metrics,
                        is_best=is_best
                    )
                    
                    if train_logger is not None and val_logger is not None:
                        if is_best:
                            train_logger.log(f"✓ 本次验证刷新最佳模型: {config.best_metric}={current_metric:.4f}，已更新 best_model.pt")
                            val_logger.log(f"✓ 本次验证刷新最佳模型: {config.best_metric}={current_metric:.4f}，已更新 best_model.pt", echo=False)
                        else:
                            train_logger.log(f"本次验证未刷新最佳模型: {config.best_metric}={current_metric:.4f}, 当前最佳={best_metric:.4f}")
                            val_logger.log(f"本次验证未刷新最佳模型: {config.best_metric}={current_metric:.4f}, 当前最佳={best_metric:.4f}", echo=False)

            if is_distributed:
                dist.barrier()  # 验证后同步：确保主进程更新保存完模型后，其余卡才能继续往前走，防止死锁
        
        if train_logger is not None:
            train_logger.log(f"Epoch {epoch + 1}/{config.num_epochs} - Train Loss: {train_loss:.4f}")
    
    if is_main_process:
        print("-" * 60)
        print(f"✓ 训练完成，最佳模型: {os.path.join(config.checkpoint_dir, 'best_model.pt')}")
        print(f"✓ 最新训练状态: {os.path.join(config.checkpoint_dir, 'last_state.pt')}")
        if train_logger is not None and val_logger is not None:
            train_logger.log("Training completed!")
            val_logger.log("Validation logging completed!")
            train_logger.save_and_close()
            val_logger.save_and_close()

    if is_distributed:
        dist.barrier()


def train_epoch(model, dataloader, optimizer, device, config, epoch, logger):
    """
    训练单个epoch
    """
    set_model_train_mode(model)
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 创建输入字典
        input_data = {
            'blur': batch['blur'].to(device),
            'sharp': batch['sharp'].to(device),
            'mask': batch['mask'].to(device),
            'img_path': batch['img_path']
        }

        model.set_input(input_data)
        model.optimize_parameters()
        
        # 计算损失
        loss_dict = model.get_current_errors()
        batch_loss = sum(v.item() if torch.is_tensor(v) else float(v) for v in loss_dict.values())
        total_loss += batch_loss
        num_batches += 1
        
        # 定期输出日志
        if logger is not None and (batch_idx + 1) % config.print_freq == 0:
            logger.log(f"  Epoch {epoch + 1} [{batch_idx + 1}/{len(dataloader)}] - Loss: {total_loss / num_batches:.6f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device, config, epoch, train_logger, val_logger, metric_calc):
    """
    验证函数
    """
    print(f"\n验证 Epoch {epoch + 1}...")
    set_model_eval_mode(model)
    
    psnr_list = []
    ssim_list = []
    loss_list = []
    
    for batch_idx, batch in enumerate(dataloader):
        # 创建输入字典
        input_data = {
            'blur': batch['blur'].to(device),
            'sharp': batch['sharp'].to(device),
            'mask': batch['mask'].to(device),
            'img_path': batch['img_path']
        }

        model.set_input(input_data)
        model.test()
        
        # 获取当前验证 batch 的验证损失
        loss_dict = model.get_current_errors()
        if loss_dict:
            loss_list.append(sum(v.item() if torch.is_tensor(v) else float(v) for v in loss_dict.values()))
        
        # 获取输出
        output = model.img_out  # [-1, 1]
        target = model.img_truth  # [-1, 1]
        
        # 转换为[0, 1]用于计算指标
        output_np = ((output[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
        target_np = ((target[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
        
        # 转换为uint8
        output_uint8 = (output_np * 255).astype(np.uint8)
        target_uint8 = (target_np * 255).astype(np.uint8)
        
        # 使用与测试阶段一致的同一套指标计算实现
        psnr, ssim = metric_calc.calculate_psnr_ssim(output_uint8, target_uint8)
        
        # 异常值防护
        if np.isfinite(psnr):
            psnr_list.append(psnr)
        if np.isfinite(ssim):
            ssim_list.append(ssim)
        
        # 保存验证可视化结果
        if config.save_val_visual and batch_idx < 5:  # 只保存前5张
            save_visual_result(output_uint8, target_uint8, batch['name'][0], config.val_visual_dir, epoch)
    
    # 计算平均指标
    avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
    avg_loss = np.mean(loss_list) if (loss_list and len(loss_list) > 0) else 0.0
    
    metrics = {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'loss': avg_loss
    }

    print(f"当前验证指标 - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LOSS: {avg_loss:.4f}")
    if val_logger is not None:
        val_logger.log_validation(epoch, metrics=metrics)
    if train_logger is not None:
        train_logger.log(f"验证结果 - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    
    set_model_train_mode(model)
    return metrics


def test(config, device):
    """单卡测试入口，直接复用 test.py 中的定量评估实现。"""
    from test import run_test
    return run_test(config, device=device)


def save_visual_result(output, target, name, save_dir, epoch):
    """保存可视化结果"""
    # 转换为BGR格式保存
    output_bgr = output[:, :, ::-1]
    target_bgr = target[:, :, ::-1]
    
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(epoch_dir, f'{name[:-4]}_output.png'), output_bgr)
    cv2.imwrite(os.path.join(epoch_dir, f'{name[:-4]}_target.png'), target_bgr)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AACNet 盲元补完网络')
    
    # 运行模式
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--test', action='store_true', help='测试模式')
    
    # 配置文件路径
    parser.add_argument('--config_path', type=str, default='./experiment.cfg', 
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    # 验证模式
    if not args.train and not args.test:
        parser.print_help()
        print("\n请指定运行模式: --train 或 --test")
        sys.exit(1)
    
    if args.train and args.test:
        print("错误: 不能同时指定 --train 和 --test")
        sys.exit(1)
    
    # 加载配置
    print(f"加载配置文件: {args.config_path}")
    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件不存在 - {args.config_path}")
        sys.exit(1)
    
    config_loader = ConfigLoader(args.config_path)
    config = config_loader.get_config()
    
    # 打印配置
    config_loader.print_config()

    # 规范 GPU id
    try:
        import torch
        if torch.cuda.is_available():
            visible_cnt = torch.cuda.device_count()
            if isinstance(config.gpu_ids, (list, tuple)) and len(config.gpu_ids) > visible_cnt:
                print(f"警告: 配置中 gpu_ids={config.gpu_ids}，但当前可见 GPU 数量为 {visible_cnt}，将截断为前 {visible_cnt} 个 id。")
                config.gpu_ids = config.gpu_ids[:visible_cnt]
        else:
            config.gpu_ids = []
    except Exception:
        pass
    
    # 初始化分布式环境（如果通过 torchrun 启动）
    init_distributed(config)

    # 设置随机种子
    setup_seed(config)
    
    # 设置设备
    device = setup_device(config)
    
    # 创建必要的目录
    setup_directories(config)

    # 运行训练或测试
    try:
        if args.train:
            train(config, device)
        else:
            test(config, device)
    finally:
        if getattr(config, 'distributed', False) and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()