#!/usr/bin/env python3
"""
AACNet 盲元测试启动脚本
支持交互式配置或命令行参数
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description='AACNet 盲元补完测试启动器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例：
  # 基础使用
  python launch_blind_test.py \\
    --data_root /home/student_server/Qtt/NAFNet/data \\
    --checkpoint /path/to/model.pth

  # 完整配置
  python launch_blind_test.py \\
    --data_root /home/student_server/Qtt/NAFNet/data \\
    --checkpoint /path/to/model.pth \\
    --save_dir ./results/test_v1 \\
    --device cuda \\
    --gpu_ids 0,1
    
  # 交互模式（无参数）
  python launch_blind_test.py
        '''
    )
    
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据集根目录')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型检查点路径')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='结果保存目录（默认自动生成）')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备：cuda 或 cpu')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU ID，多个用逗号分隔')
    parser.add_argument('--image_border', type=int, default=0,
                        help='PSNR/SSIM计算时的边界裁剪像素')
    parser.add_argument('--non-interactive', action='store_true',
                        help='非交互模式（必须提供--data_root和--checkpoint）')
    
    args = parser.parse_args()
    
    # 交互模式
    if not args.non_interactive and (args.data_root is None or args.checkpoint is None):
        print("\n" + "="*60)
        print("AACNet 盲元补完测试 - 交互配置")
        print("="*60 + "\n")
        
        # 获取数据集路径
        while not args.data_root:
            data_root = input("请输入数据集根目录路径: ").strip()
            if os.path.isdir(data_root):
                args.data_root = data_root
                print(f"✓ 数据集目录: {data_root}")
                
                # 检查子目录
                if os.path.isdir(os.path.join(data_root, 'test_blur')):
                    print("  ✓ 找到 test_blur")
                if os.path.isdir(os.path.join(data_root, 'test_sharp')):
                    print("  ✓ 找到 test_sharp")
                if os.path.isdir(os.path.join(data_root, 'test_mask')):
                    print("  ✓ 找到 test_mask")
            else:
                print("✗ 目录不存在，请重试")
        
        # 获取模型路径
        while not args.checkpoint:
            checkpoint = input("\n请输入模型检查点路径: ").strip()
            if os.path.isfile(checkpoint):
                args.checkpoint = checkpoint
                print(f"✓ 模型文件: {checkpoint}")
            else:
                response = input(f"⚠ 文件不存在: {checkpoint}\n是否继续？(y/n): ").lower()
                if response == 'y':
                    args.checkpoint = checkpoint
                    break
        
        # 获取保存目录（可选）
        if not args.save_dir:
            default_save = f"./results/aacnet_blind_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            save_dir = input(f"\n请输入结果保存目录 (默认: {default_save}): ").strip()
            args.save_dir = save_dir if save_dir else default_save
        
        # 获取GPU设置
        gpu_input = input(f"\n请输入GPU ID (默认: {args.gpu_ids}): ").strip()
        if gpu_input:
            args.gpu_ids = gpu_input
        
        # 设备确认
        device_input = input(f"\n请选择设备 (默认: {args.device}, 输入 'cpu' 使用CPU): ").strip().lower()
        if device_input:
            args.device = device_input
        
        print("\n" + "="*60)
        print("配置确认")
        print("="*60)
        print(f"数据集: {args.data_root}")
        print(f"模型: {args.checkpoint}")
        print(f"输出: {args.save_dir}")
        print(f"设备: {args.device}")
        print(f"GPU: {args.gpu_ids}")
        print("="*60 + "\n")
        
        confirm = input("是否开始测试？(y/n): ").lower()
        if confirm != 'y':
            print("已取消")
            sys.exit(0)
    
    elif args.non_interactive or (args.data_root and args.checkpoint):
        # 验证必需参数
        if not args.data_root:
            print("ERROR: --data_root 不能为空")
            sys.exit(1)
        if not args.checkpoint:
            print("ERROR: --checkpoint 不能为空")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    
    # 自动生成保存目录（如果未指定）
    if not args.save_dir:
        args.save_dir = f"./results/aacnet_blind_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 构建命令
    cmd = [
        'python', 'test_blind_aacnet.py',
        '--data_root', args.data_root,
        '--checkpoint', args.checkpoint,
        '--save_dir', args.save_dir,
        '--device', args.device,
        '--gpu_ids', args.gpu_ids,
        '--image_border', str(args.image_border),
        '--model', 'aacnet',
        '--name', 'aacnet_blind'
    ]
    
    print("\n" + "="*60)
    print("执行命令:")
    print(" ".join(cmd))
    print("="*60 + "\n")
    
    # 运行测试
    try:
        result = subprocess.run(cmd, check=True)
        
        print("\n" + "="*60)
        print("✓ 测试完成！")
        print("="*60)
        print(f"\n结果保存位置: {args.save_dir}")
        print("\n输出内容:")
        print(f"  - 补完图像: {args.save_dir}/test/")
        print(f"  - 评估指标: {args.save_dir}/blind_eval/")
        
        # 显示指标文件
        metrics_file = os.path.join(args.save_dir, 'blind_eval', 'test_blind_metrics.csv')
        if os.path.isfile(metrics_file):
            print(f"\n✓ 全局指标已保存至: {metrics_file}")
            print("\n前几行指标:")
            with open(metrics_file, 'r') as f:
                for i, line in enumerate(f):
                    if i < 3:
                        print(f"  {line.rstrip()}")
                    else:
                        break
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 测试失败（返回码: {e.returncode}）")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n已被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())
