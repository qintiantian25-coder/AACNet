#!/bin/bash
# AACNet 盲元测试快速启动脚本示例

# ============================================
# 配置部分 - 修改这些参数
# ============================================

# 数据集路径（使用绝对路径）
DATA_ROOT="/home/student_server/Qtt/NAFNet/data"

# 模型检查点路径
CHECKPOINT="/home/tianyu/Pythonproject/AACNet/checkpoints/aacnet_blind/net_G_latest.pth"

# 结果保存目录
SAVE_DIR="./results/aacnet_blind_test_$(date +%Y%m%d_%H%M%S)"

# GPU设置（单GPU或多GPU）
GPU_IDS="0"

# 其他参数
IMAGE_BORDER=0              # PSNR/SSIM计算时的边界裁剪像素数
DEVICE="cuda"               # cuda 或 cpu

# ============================================
# 检查依赖和数据集
# ============================================

echo "======================================"
echo "AACNet 盲元补完测试"
echo "======================================"

# 检查数据集
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: 数据集目录不存在: $DATA_ROOT"
    exit 1
fi

echo "✓ 数据集目录: $DATA_ROOT"

# 检查测试数据
if [ ! -d "$DATA_ROOT/test_blur" ]; then
    echo "ERROR: test_blur 目录不存在"
    exit 1
fi

echo "✓ 测试集 (test_blur) 存在"

# 检查模型
if [ ! -f "$CHECKPOINT" ]; then
    echo "WARNING: 模型检查点不存在: $CHECKPOINT"
    echo "         使用随机初始化模型运行测试"
fi

# ============================================
# 运行测试
# ============================================

echo ""
echo "开始测试..."
echo "结果将保存到: $SAVE_DIR"
echo ""

cd "$(dirname "$0")"

python test_blind_aacnet.py \
    --data_root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --save_dir "$SAVE_DIR" \
    --device "$DEVICE" \
    --gpu_ids "$GPU_IDS" \
    --image_border "$IMAGE_BORDER" \
    --model aacnet \
    --name aacnet_blind

# ============================================
# 结果总结
# ============================================

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✓ 测试完成！"
    echo "======================================"
    echo "结果位置: $SAVE_DIR"
    echo ""
    echo "输出文件："
    echo "  - 补完结果: $SAVE_DIR/test/"
    echo "  - 评估指标: $SAVE_DIR/blind_eval/"
    echo ""
    
    # 显示全局指标
    if [ -f "$SAVE_DIR/blind_eval/test_blind_metrics.csv" ]; then
        echo "全局指标汇总:"
        head -5 "$SAVE_DIR/blind_eval/test_blind_metrics.csv"
    fi
else
    echo ""
    echo "ERROR: 测试失败！"
    exit 1
fi
