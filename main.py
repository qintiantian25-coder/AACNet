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

# 导入项目模块
from util.config_loader import ConfigLoader
from dataloader.blind_pixel_loader import create_dataloader
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
        torch.cuda.manual_seed_all(config.seed)
    
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
    
    if config.deterministic:
        torch.backends.cudnn.deterministic = True


def setup_device(config):
    """设置计算设备"""
    if torch.cuda.is_available() and len(config.gpu_ids) > 0:
        device = torch.device(f'cuda:{config.gpu_ids[0]}')
        print(f"Using GPU: {config.gpu_ids}")
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
    
    # 创建数据加载器
    print("\n正在加载数据...")
    train_loader = create_dataloader(config, phase='train', shuffle=True)
    val_loader = create_dataloader(config, phase='val', shuffle=False)
    
    print(f"  训练集大小: {len(train_loader.dataset)}")
    print(f"  验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    print("\n正在构建模型...")
    # create_model期望opt.model属性，所以设置它
    config.model = config.model_name
    # 添加模型期望的其他属性
    config.lr = config.learning_rate
    config.isTrain = True
    # 添加checkpoint相关的参数
    if not hasattr(config, 'checkpoints_dir'):
        config.checkpoints_dir = config.checkpoint_dir
    if not hasattr(config, 'name'):
        config.name = config.model_prefix
    if not hasattr(config, 'which_iter'):
        config.which_iter = 0
    if not hasattr(config, 'continue_train'):
        config.continue_train = False
    
    model = create_model(config)
    model.to(device)
    if len(config.gpu_ids) > 1 and config.use_dataparallel:
        model = nn.DataParallel(model, device_ids=config.gpu_ids)
    print(f"  模型: {config.model_name}")
    
    # 创建检查点管理器
    checkpoint_manager = CheckpointManager(config, config.best_metric)
    
    # 创建日志记录器
    train_logger = Logger(config, log_name='training')
    val_logger = Logger(config, log_name='validation')
    train_logger.log_config(config)
    
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
        # 自动寻找最新的训练状态文件
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
        # 训练阶段
        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch, train_logger)
        
        # 获取当前学习率
        if hasattr(model, 'schedulers') and len(model.schedulers) > 0:
            model.update_learning_rate()
            current_lr = model.optimizers[0].param_groups[0]['lr']
            train_logger.log(f"Epoch {epoch + 1}/{config.num_epochs} - LR: {current_lr:.6f}")

        # 保存当前训练状态到单一检查点文件，便于随时续训
        checkpoint_manager.save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            {'loss': train_loss},
            is_best=False
        )
        
        # 验证阶段（每val_interval轮进行一次）
        if (epoch + 1) % config.val_interval == 0:
            val_metrics = validate(model, val_loader, device, config, epoch, train_logger, val_logger, metric_calc)
            
            # 检查是否是最好的模型
            current_metric = val_metrics[config.best_metric]
            
            if best_metric is None:
                best_metric = current_metric
                is_best = True
            elif config.best_metric == 'psnr':
                # PSNR越高越好
                is_best = current_metric > best_metric
                if is_best:
                    best_metric = current_metric
            else:
                # 其他指标（如loss）越低越好
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
                
                if is_best:
                    train_logger.log(f"✓ 本次验证刷新最佳模型: {config.best_metric}={current_metric:.4f}，已更新 best_model.pt")
                    val_logger.log(f"✓ 本次验证刷新最佳模型: {config.best_metric}={current_metric:.4f}，已更新 best_model.pt")
                else:
                    train_logger.log(f"本次验证未刷新最佳模型: {config.best_metric}={current_metric:.4f}, 当前最佳={best_metric:.4f}")
                    val_logger.log(f"本次验证未刷新最佳模型: {config.best_metric}={current_metric:.4f}, 当前最佳={best_metric:.4f}")
        
        train_logger.log(f"Epoch {epoch + 1}/{config.num_epochs} - Train Loss: {train_loss:.4f}")
    
    print("-" * 60)
    print(f"✓ 训练完成，单一检查点文件: {os.path.join(config.checkpoint_dir, f'{config.model_prefix}.pt')}（包含最新训练状态与最佳模型权重）")
    train_logger.log("Training completed!")
    val_logger.log("Validation logging completed!")
    train_logger.save_and_close()
    val_logger.save_and_close()


def train_epoch(model, dataloader, optimizer, device, config, epoch, logger):
    """
    训练单个epoch
    
    Args:
        model: 模型
        dataloader: 训练数据加载器
        optimizer: 优化器（未使用，模型内部有自己的优化器）
        device: 计算设备
        config: 配置
        epoch: 当前epoch
        logger: 日志记录器
    
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 创建输入字典
        input_data = {
            'img': batch['blur'].to(device),
            'mask': batch['mask'].to(device),
            'img_path': batch['img_path']
        }
        
        # 扩展mask到3通道
        if input_data['mask'].shape[1] == 1:
            input_data['mask'] = input_data['mask'].repeat(1, 3, 1, 1)
        
        # 设置输入并优化参数
        model.set_input(input_data)
        model.optimize_parameters()
        
        # 计算损失
        loss_dict = model.get_current_errors()
        total_loss += sum(loss_dict.values())
        num_batches += 1
        
        # 定期输出日志
        if (batch_idx + 1) % config.print_freq == 0:
            logger.log(f"  Epoch {epoch + 1} [{batch_idx + 1}/{len(dataloader)}] - Loss: {total_loss / num_batches:.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device, config, epoch, train_logger, val_logger, metric_calc):
    """
    验证函数
    
    Args:
        model: 模型
        dataloader: 验证数据加载器
        device: 计算设备
        config: 配置
        epoch: 当前epoch
        logger: 日志记录器
        metric_calc: 指标计算器
    
    Returns:
        metrics: 验证指标字典
    """
    print(f"\n验证 Epoch {epoch + 1}...")
    model.eval()
    
    psnr_list = []
    ssim_list = []
    loss_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 创建输入字典
            input_data = {
                'img': batch['blur'].to(device),
                'mask': batch['mask'].to(device),
                'img_path': batch['img_path']
            }
            
            # 扩展mask到3通道
            if input_data['mask'].shape[1] == 1:
                input_data['mask'] = input_data['mask'].repeat(1, 3, 1, 1)
            
            # 前向传播
            model.set_input(input_data)
            model.test()
            
            # 获取输出
            output = model.img_out  # [-1, 1]
            target = model.img_truth  # [-1, 1]
            
            # 转换为[0, 1]用于计算指标
            output_np = ((output[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
            target_np = ((target[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
            
            # 转换为uint8
            output_uint8 = (output_np * 255).astype(np.uint8)
            target_uint8 = (target_np * 255).astype(np.uint8)
            
            # 计算PSNR和SSIM
            psnr = metric_calc.calculate_psnr(output_uint8, target_uint8)
            ssim = metric_calc.calculate_ssim(output_uint8, target_uint8)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            # 保存验证可视化结果
            if config.save_val_visual and batch_idx < 5:  # 只保存前5张
                save_visual_result(output_uint8, target_uint8, batch['name'][0], config.val_visual_dir, epoch)
    
    # 计算平均指标
    avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
    
    print(f"当前验证指标 - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}, LOSS: {metrics['loss']:.4f}")
    val_logger.log_validation(epoch, metrics={
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'loss': np.mean(loss_list) if loss_list else 0.0
    })
    train_logger.log(f"验证结果 - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    
    metrics = {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'loss': np.mean(loss_list) if loss_list else 0.0
    }
    
    model.train()
    return metrics


def test(config, device):
    """
    测试函数
    
    Args:
        config: 配置对象
        device: 计算设备
    """
    print("\n" + "="*60)
    print("开始测试 (Testing)")
    print("="*60)
    
    # 创建数据加载器
    print("\n正在加载测试数据...")
    test_loader = create_dataloader(config, phase='test', shuffle=False)
    print(f"  测试集大小: {len(test_loader.dataset)}")
    
    # 创建模型
    print("\n正在加载模型...")
    # create_model期望opt.model属性，所以设置它
    config.model = config.model_name
    # 添加模型期望的其他属性
    config.lr = config.learning_rate
    config.isTrain = False
    # 添加checkpoint相关的参数
    if not hasattr(config, 'checkpoints_dir'):
        config.checkpoints_dir = config.checkpoint_dir
    if not hasattr(config, 'name'):
        config.name = config.model_prefix
    if not hasattr(config, 'which_iter'):
        config.which_iter = 0
    if not hasattr(config, 'continue_train'):
        config.continue_train = False
    
    model = create_model(config)
    model.to(device)
    model.eval()
    
    # 加载最佳检查点
    checkpoint_manager = CheckpointManager(config, config.best_metric)
    best_ckpt = checkpoint_manager.find_best_checkpoint()
    
    if best_ckpt:
        print(f"加载 best_model.pt 中的最佳模型权重: {best_ckpt}")
        checkpoint_manager.load_checkpoint(best_ckpt, model, load_weights_only=True, use_best=True)
    else:
        print("警告: 未找到 best_model.pt，使用随机初始化的模型")
    
    # 创建指标计算器
    metric_calc = MetricCalculator(config.crop_border)
    
    # 创建结果保存目录
    os.makedirs(config.results_dir, exist_ok=True)
    
    # 测试循环
    print(f"\n开始测试...")
    print("-" * 60)
    
    results = []
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 创建输入字典
            input_data = {
                'img': batch['blur'].to(device),
                'mask': batch['mask'].to(device),
                'img_path': batch['img_path']
            }
            
            # 扩展mask到3通道
            if input_data['mask'].shape[1] == 1:
                input_data['mask'] = input_data['mask'].repeat(1, 3, 1, 1)
            
            # 前向传播
            model.set_input(input_data)
            model.test()
            
            # 获取输出
            output = model.img_out  # [-1, 1]
            target = model.img_truth  # [-1, 1]
            
            # 转换为[0, 1]用于计算指标
            output_np = ((output[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
            target_np = ((target[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
            
            # 转换为uint8
            output_uint8 = (output_np * 255).astype(np.uint8)
            target_uint8 = (target_np * 255).astype(np.uint8)
            
            # 计算指标
            psnr = metric_calc.calculate_psnr(output_uint8, target_uint8)
            ssim = metric_calc.calculate_ssim(output_uint8, target_uint8)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            # 保存结果
            if config.save_results:
                # 转换回BGR格式保存
                output_bgr = (output_np[:, :, ::-1] * 255).astype(np.uint8)
                result_path = os.path.join(config.results_dir, batch['name'][0])
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                cv2.imwrite(result_path, output_bgr)
            
            # 记录结果
            results.append({
                'image': batch['name'][0],
                'group': batch['group'][0],
                'psnr': psnr,
                'ssim': ssim
            })
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  处理 [{batch_idx + 1}/{len(test_loader)}] - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
    
    # 保存测试结果到CSV
    csv_path = os.path.join(config.results_dir, 'test_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'group', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(results)
    
    # 计算平均指标
    avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
    
    print("-" * 60)
    print(f"\n测试完成!")
    print(f"  平均 PSNR: {avg_psnr:.4f}")
    print(f"  平均 SSIM: {avg_ssim:.4f}")
    print(f"  结果保存到: {config.results_dir}")
    print(f"  指标保存到: {csv_path}")


def save_visual_result(output, target, name, save_dir, epoch):
    """保存可视化结果"""
    import cv2
    
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
    
    # 设置随机种子
    setup_seed(config)
    
    # 设置设备
    device = setup_device(config)
    
    # 创建必要的目录
    setup_directories(config)

    # 确保检查点目录中只保留单一 best_model.pt，历史遗留文件已经在 CheckpointManager 中清理
    
    # 运行训练或测试
    if args.train:
        train(config, device)
    else:
        test(config, device)


if __name__ == '__main__':
    main()
