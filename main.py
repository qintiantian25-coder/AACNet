"""
AACNet 盲元补完网络 - 主训练和测试脚本

用法:
    训练: torchrun --nproc_per_node=2 main.py --train --config_path ./experiment.cfg
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
        # NOTE: 不在此处调用全局的 CUDA seed，避免提前触发对 cuda:0 的上下文创建。
    
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = False
    
    if config.deterministic:
        torch.backends.cudnn.deterministic = True


def setup_device(config):
    """设置计算设备并为当前卡独立绑定 CUDA 种子"""
    if torch.cuda.is_available() and len(config.gpu_ids) > 0:
        if getattr(config, 'distributed', False):
            torch.cuda.set_device(config.local_rank)
            device = torch.device(f'cuda:{config.local_rank}')
        else:
            torch.cuda.set_device(config.gpu_ids[0])
            device = torch.device(f'cuda:{config.gpu_ids[0]}')
            
        # 精准对本进程绑定的单卡设置 CUDA 种子，拒绝干扰其他卡
        if getattr(config, 'seed', -1) != -1 and device.type == 'cuda':
            try:
                torch.cuda.manual_seed(config.seed)
            except Exception:
                pass
                
        if getattr(config, 'distributed', False):
            print(f"[Rank {config.rank}] DDP 已绑定到本地设备: {torch.cuda.current_device()}，可见 GPU: {config.gpu_ids}")
        else:
            print(f"单卡模式使用 GPU: {config.gpu_ids}")
    else:
        device = torch.device('cpu')
        print("使用 CPU 模式")
    
    return device


def setup_directories(config):
    """创建必要的目录（仅在主进程中物理创建）"""
    if getattr(config, 'is_main_process', True):
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        if config.save_val_visual:
            os.makedirs(config.val_visual_dir, exist_ok=True)
        if hasattr(config, 'save_results') and config.save_results:
            os.makedirs(config.results_dir, exist_ok=True)


def set_model_train_mode(model):
    """设置模型至训练状态"""
    if hasattr(model, 'net_G'):
        model.net_G.train()
    elif hasattr(model, 'train'):
        model.train()


def set_model_eval_mode(model):
    """设置模型至评估状态"""
    if hasattr(model, 'net_G'):
        model.net_G.eval()
    elif hasattr(model, 'eval'):
        model.eval()


def init_distributed(config):
    """初始化 DDP 分布式环境"""
    local_rank_env = os.environ.get('LOCAL_RANK')
    world_size_env = os.environ.get('WORLD_SIZE')
    rank_env = os.environ.get('RANK')

    distributed = local_rank_env is not None and world_size_env is not None and int(world_size_env) > 1
    config.distributed = distributed
    config.local_rank = int(local_rank_env) if local_rank_env is not None else 0
    config.rank = int(rank_env) if rank_env is not None else 0
    config.world_size = int(world_size_env) if world_size_env is not None else 1
    config.is_main_process = (config.rank == 0)

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP 需要 PyTorch 具备 CUDA 环境支持')
        torch.cuda.set_device(config.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.empty_cache()
        
        # 分布式状态下，将配置中当前可见单卡修正为当前 Local Rank
        config.gpu_ids = [config.local_rank]
        config.use_dataparallel = False

    return distributed


def train(config, device):
    """核心训练函数"""
    is_main_process = getattr(config, 'is_main_process', True)
    is_distributed = getattr(config, 'distributed', False)

    if is_main_process:
        print("\n" + "="*60)
        print("开始初始化训练流程 (Training Init)")
        print("="*60)
    
    # 1. 创建数据加载器
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
    
    if is_main_process:
        print(f" -> 训练集总样本数: {len(train_loader.dataset)}")
        if val_loader is not None:
            print(f" -> 验证集总样本数: {len(val_loader.dataset)}")
    
    # 2. 补全框架参数配置
    config.model = config.model_name
    config.lr = config.learning_rate
    config.isTrain = True
    if not hasattr(config, 'checkpoints_dir'):
        config.checkpoints_dir = config.checkpoint_dir
    if not hasattr(config, 'name'):
        config.name = config.model_prefix
    if not hasattr(config, 'which_iter'):
        config.which_iter = 0
    if not hasattr(config, 'continue_train'):
        config.continue_train = False
    
    # 3. 【绝对防御方案】强制在 CPU 上安全构建模型
    if is_main_process:
        print("\n正在通过 CPU 隔离区安全构建基础网络结构...")
    
    real_gpu_ids = config.gpu_ids  # 暂存真实的 GPU 拓扑
    config.gpu_ids = []            # 诱导框架内部将结构全部建立在 CPU 内存上
    
    with torch.no_grad():
        model = create_model(config)
    
    config.gpu_ids = real_gpu_ids  # 恢复配置
    
    # 4. 精准搬运子网络到分配的单卡，随后安全激活优化器
    if hasattr(model, 'net_G'):
        model.net_G = model.net_G.to(device)
        # 激活优化器和设置逻辑：传入精确绑定的当前单卡配置 [local_rank] 
        config.gpu_ids = [config.local_rank] if is_distributed else real_gpu_ids
        if hasattr(model, 'setup'):
            model.setup(config)

    # 5. DDP 分布式并行包裹
    if is_distributed and config.world_size > 1 and hasattr(model, 'net_G'):
        model.net_G = DDP(
            model.net_G,
            device_ids=[config.local_rank],
            output_device=config.local_rank,
            find_unused_parameters=False,
            broadcast_buffers=False,
        )
        
    if is_main_process:
        print(f" -> 网络基础模型 [{config.model_name}] 构建并迁移单卡设备成功。")
    
    # 创建检查点管理器与日志记录器
    checkpoint_manager = CheckpointManager(config, config.best_metric)
    train_logger = Logger(config, log_name='training') if is_main_process else None
    val_logger = Logger(config, log_name='validation') if is_main_process else None
    if train_logger is not None:
        train_logger.log_config(config)

    if is_distributed:
        dist.barrier()
    
    # 恢复训练检查点
    start_epoch = 0
    if config.resume_training and config.checkpoint_path:
        if is_main_process:
            print(f"\n从指定路径恢复训练状态: {config.checkpoint_path}")
        start_epoch = checkpoint_manager.load_checkpoint(
            config.checkpoint_path, model, load_weights_only=config.load_weights_only, use_best=False
        )
    elif config.resume_training:
        latest_ckpt = checkpoint_manager.find_latest_checkpoint()
        if latest_ckpt:
            if is_main_process:
                print(f"\n自动检索并从最新的训练状态恢复: {latest_ckpt}")
            start_epoch = checkpoint_manager.load_checkpoint(
                latest_ckpt, model, load_weights_only=config.load_weights_only, use_best=False
            )
    
    if is_main_process:
        print(f"\n★ 开始循环迭代训练 ({start_epoch}/{config.num_epochs} epochs) ★")
        print("-" * 60)
    
    best_metric = None
    metric_calc = MetricCalculator(config.crop_border)
    optimizer = getattr(model, 'optimizers', [None])[0] if hasattr(model, 'optimizers') and len(getattr(model, 'optimizers', [])) > 0 else None
    scheduler = getattr(model, 'schedulers', [None])[0] if hasattr(model, 'schedulers') and len(getattr(model, 'schedulers', [])) > 0 else None
    
    for epoch in range(start_epoch, config.num_epochs):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # 执行单轮训练
        train_loss = train_epoch(model, train_loader, optimizer, device, config, epoch, train_logger)

        # 多卡损失同步聚合
        if is_distributed:
            loss_tensor = torch.tensor(train_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            train_loss = loss_tensor.item() / config.world_size
        
        # 更新并获取当前学习率
        if hasattr(model, 'schedulers') and len(model.schedulers) > 0:
            model.update_learning_rate()
            if len(model.optimizers) > 0:
                current_lr = model.optimizers[0].param_groups[0]['lr']
                if train_logger is not None:
                    train_logger.log(f"Epoch {epoch + 1}/{config.num_epochs} - LR: {current_lr:.6e}")

        # 主进程保存最新的断点检查点
        if is_main_process:
            checkpoint_manager.save_checkpoint(
                model, optimizer, scheduler, epoch + 1, {'loss': train_loss}, is_best=False
            )
        
        # 定期进行验证评估
        if (epoch + 1) % config.val_interval == 0:
            if is_distributed:
                dist.barrier()

            val_metrics = validate(model, val_loader, device, config, epoch, train_logger, val_logger, metric_calc) if is_main_process else None
            
            if is_main_process and val_metrics is not None:
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
                
                if is_best or not config.save_best_only:
                    checkpoint_manager.save_checkpoint(
                        model, optimizer, scheduler, epoch + 1, val_metrics, is_best=is_best
                    )
                    
                    if train_logger is not None and val_logger is not None:
                        if is_best:
                            train_logger.log(f"✓ 本次验证刷新最佳纪录: {config.best_metric}={current_metric:.4f}，已更新 best_model.pt")
                            val_logger.log(f"✓ 本次验证刷新最佳纪录: {config.best_metric}={current_metric:.4f}，已更新 best_model.pt")
                        else:
                            train_logger.log(f"验证未刷新纪录: 当前 {config.best_metric}={current_metric:.4f}, 历史最佳={best_metric:.4f}")
                            val_logger.log(f"验证未刷新纪录: 当前 {config.best_metric}={current_metric:.4f}, 历史最佳={best_metric:.4f}")

            if is_distributed:
                dist.barrier()
        
        if train_logger is not None:
            train_logger.log(f"Epoch {epoch + 1}/{config.num_epochs} - Avg Train Loss: {train_loss:.6f}")
    
    if is_main_process:
        print("-" * 60)
        print(f"✓ 训练全部跑完！模型检查点及日志已被安全储存。")
        if train_logger is not None and val_logger is not None:
            train_logger.save_and_close()
            val_logger.save_and_close()

    if is_distributed:
        dist.barrier()


def train_epoch(model, dataloader, optimizer, device, config, epoch, logger):
    """训练单轮 Epoch"""
    set_model_train_mode(model)
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_data = {
            'img': batch['blur'].to(device, non_blocking=True),
            'mask': batch['mask'].to(device, non_blocking=True),
            'img_path': batch['img_path']
        }
        
        if input_data['mask'].shape[1] == 1:
            input_data['mask'] = input_data['mask'].repeat(1, 3, 1, 1)
        
        model.set_input(input_data)
        model.optimize_parameters()
        
        loss_dict = model.get_current_errors()
        total_loss += sum(loss_dict.values())
        num_batches += 1
        
        if logger is not None and (batch_idx + 1) % config.print_freq == 0:
            logger.log(f"  Epoch {epoch + 1} [{batch_idx + 1}/{len(dataloader)}] - Iter Loss: {sum(loss_dict.values()):.6f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, device, config, epoch, train_logger, val_logger, metric_calc):
    """主进程验证集指标推断"""
    print(f"\n正在对验证集执行指标评估 (Epoch {epoch + 1})...")
    set_model_eval_mode(model)
    
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_data = {
                'img': batch['blur'].to(device),
                'mask': batch['mask'].to(device),
                'img_path': batch['img_path']
            }
            
            if input_data['mask'].shape[1] == 1:
                input_data['mask'] = input_data['mask'].repeat(1, 3, 1, 1)
            
            model.set_input(input_data)
            model.test()
            
            output = model.img_out   # [-1, 1]
            target = model.img_truth # [-1, 1]
            
            output_np = ((output[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
            target_np = ((target[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
            
            output_uint8 = (np.clip(output_np * 255, 0, 255)).astype(np.uint8)
            target_uint8 = (np.clip(target_np * 255, 0, 255)).astype(np.uint8)
            
            psnr = metric_calc.calculate_psnr(output_uint8, target_uint8)
            ssim = metric_calc.calculate_ssim(output_uint8, target_uint8)
            
            if np.isfinite(psnr):
                psnr_list.append(psnr)
            if np.isfinite(ssim):
                ssim_list.append(ssim)
            
            if config.save_val_visual and batch_idx < 5:
                save_visual_result(output_uint8, target_uint8, batch['name'][0], config.val_visual_dir, epoch)
    
    avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
    
    metrics = {'psnr': avg_psnr, 'ssim': avg_ssim, 'loss': 0.0}

    print(f" -> 验证完成 | 平均 PSNR: {avg_psnr:.4f}, 平均 SSIM: {avg_ssim:.4f}")
    if val_logger is not None:
        val_logger.log_validation(epoch, metrics=metrics)
    
    set_model_train_mode(model)
    return metrics


def test(config, device):
    """单卡独立测试推断"""
    print("\n" + "="*60)
    print("开始执行测试模式 (Testing Inference)")
    print("="*60)
    
    test_loader = create_dataloader(config, phase='test', shuffle=False)
    print(f" -> 测试集样本总量: {len(test_loader.dataset)}")
    
    config.model = config.model_name
    config.lr = config.learning_rate
    config.isTrain = False
    if not hasattr(config, 'checkpoints_dir'):
        config.checkpoints_dir = config.checkpoint_dir
    if not hasattr(config, 'name'):
        config.name = config.model_prefix

    # CPU 隔离构建后推进单卡设备
    real_gpu_ids = config.gpu_ids
    config.gpu_ids = []
    model = create_model(config)
    config.gpu_ids = real_gpu_ids
    
    if hasattr(model, 'net_G'):
        model.net_G = model.net_G.to(device)
        
    set_model_eval_mode(model)
    
    # 载入最优检查点权重
    checkpoint_manager = CheckpointManager(config, config.best_metric)
    best_ckpt = checkpoint_manager.find_best_checkpoint()
    
    if best_ckpt:
        print(f" -> 成功读取最佳网络权重: {best_ckpt}")
        checkpoint_manager.load_checkpoint(best_ckpt, model, load_weights_only=True, use_best=True)
    else:
        print(" [警告] 未能匹配到任何 best_model.pt 文件！将采用未训练的初始状态输出结果。")
    
    metric_calc = MetricCalculator(config.crop_border)
    os.makedirs(config.results_dir, exist_ok=True)
    
    results = []
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_data = {
                'img': batch['blur'].to(device),
                'mask': batch['mask'].to(device),
                'img_path': batch['img_path']
            }
            
            if input_data['mask'].shape[1] == 1:
                input_data['mask'] = input_data['mask'].repeat(1, 3, 1, 1)
            
            model.set_input(input_data)
            model.test()
            
            output = model.img_out
            target = model.img_truth
            
            output_np = ((output[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
            target_np = ((target[0].cpu().numpy() + 1) / 2).transpose(1, 2, 0)
            
            output_uint8 = (np.clip(output_np * 255, 0, 255)).astype(np.uint8)
            target_uint8 = (np.clip(target_np * 255, 0, 255)).astype(np.uint8)
            
            psnr = metric_calc.calculate_psnr(output_uint8, target_uint8)
            ssim = metric_calc.calculate_ssim(output_uint8, target_uint8)
            
            if np.isfinite(psnr):
                psnr_list.append(psnr)
            if np.isfinite(ssim):
                ssim_list.append(ssim)
            
            if hasattr(config, 'save_results') and config.save_results:
                output_bgr = (output_np[:, :, ::-1] * 255).astype(np.uint8)
                result_path = os.path.join(config.results_dir, batch['name'][0])
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                cv2.imwrite(result_path, output_bgr)
            
            results.append({
                'image': batch['name'][0],
                'group': batch['group'][0],
                'psnr': psnr,
                'ssim': ssim
            })
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  推断进度 [{batch_idx + 1}/{len(test_loader)}] - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
    
    csv_path = os.path.join(config.results_dir, 'test_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'group', 'psnr', 'ssim'])
        writer.writeheader()
        writer.writerows(results)
    
    print("-" * 60)
    print(f"测试圆满完成！ | 平均 PSNR: {np.mean(psnr_list):.4f}, 平均 SSIM: {np.mean(ssim_list):.4f}")
    print(f"数据汇总报表已安全保存至: {csv_path}")


def save_visual_result(output, target, name, save_dir, epoch):
    """保存可视化图片结构"""
    output_bgr = output[:, :, ::-1]
    target_bgr = target[:, :, ::-1]
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch + 1}')
    os.makedirs(epoch_dir, exist_ok=True)
    cv2.imwrite(os.path.join(epoch_dir, f'{name[:-4]}_output.png'), output_bgr)
    cv2.imwrite(os.path.join(epoch_dir, f'{name[:-4]}_target.png'), target_bgr)


def main():
    """程序统一入口"""
    parser = argparse.ArgumentParser(description='AACNet 盲元补完网络')
    parser.add_argument('--train', action='store_true', help='训练模式')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--config_path', type=str, default='./experiment.cfg', help='配置文件路径')
    args = parser.parse_args()
    
    if not args.train and not args.test:
        parser.print_help()
        print("\n[错误] 请指定运行模式: --train 或 --test")
        sys.exit(1)
    if args.train and args.test:
        print("[错误] 不能同时指定 --train 和 --test")
        sys.exit(1)
    
    if not os.path.exists(args.config_path):
        print(f"[错误] 配置文件物理路径不存在 - {args.config_path}")
        sys.exit(1)
    
    config_loader = ConfigLoader(args.config_path)
    config = config_loader.get_config()
    config_loader.print_config()

    # 1. 规范基础可见卡片截断
    try:
        if torch.cuda.is_available():
            visible_cnt = torch.cuda.device_count()
            if isinstance(config.gpu_ids, (list, tuple)) and len(config.gpu_ids) > visible_cnt:
                print(f"[警告] 配置中指定卡数过大 {config.gpu_ids}，已自适应截断为前 {visible_cnt} 张可见显卡。")
                config.gpu_ids = config.gpu_ids[:visible_cnt]
        else:
            config.gpu_ids = []
    except Exception:
        pass
    
    # 2. 率先初始化分布式拓扑(判断是否经由 torchrun 激活)
    init_distributed(config)

    # 3. 配置基本种子
    setup_seed(config)
    
    # 4. 精准获取并分配单卡专属设备
    device = setup_device(config)
    
    # 5. 主进程建立系统文件目录
    setup_directories(config)

    # 6. 分流调度
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