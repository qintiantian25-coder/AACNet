# AACNet 盲元补完测试框架

## 📋 概览

本框架为AACNet提供完整的盲元（dead pixels）补完测试能力，支持以下功能：

- ✅ 完整的盲元补完管道
- ✅ 多种盲元坐标格式（CSV、mask图像、动态坐标）
- ✅ 全面的评估指标（PSNR、SSIM、blind_mae、blind_rmse、blind_psnr）
- ✅ 自动化的结果处理和可视化
- ✅ 支持多GPU并行推理
- ✅ 交互式配置工具

## 📁 创建的文件清单

### 核心文件

| 文件 | 用途 | 说明 |
|------|------|------|
| `test_blind_aacnet.py` | **主测试脚本** | 核心测试程序，完整的盲元评估实现 |
| `model/aacnet_model.py` | **模型定义** | AACNetBlind模型类，继承BaseModel |
| `launch_blind_test.py` | **启动工具** | 交互式/命令行配置脚本 |
| `run_blind_test.sh` | **Bash启动脚本** | Shell版本的快速启动脚本 |

### 文档文件

| 文件 | 内容 |
|------|------|
| `TEST_BLIND_GUIDE.md` | 详细使用指南、参数说明、常见问题 |
| `BLIND_TEST_README.md` | 本文件，快速参考指南 |

## 🚀 快速开始

### 方式1：交互式配置（推荐新手）

```bash
cd /home/tianyu/Pythonproject/AACNet
python launch_blind_test.py
```

脚本将逐步引导你输入必要参数。

### 方式2：命令行参数

```bash
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint /path/to/model.pth \
    --save_dir ./results/test_v1
```

### 方式3：Bash脚本

```bash
bash run_blind_test.sh
```

编辑脚本顶部的参数配置部分。

## 📊 核心参数

```bash
# 必需参数
--data_root PATH           # 数据集根目录
--checkpoint PATH          # 模型权重文件

# 可选参数
--save_dir PATH            # 结果保存目录（默认自动生成）
--device DEVICE            # cuda 或 cpu（默认cuda）
--gpu_ids IDS              # GPU ID，如 0,1,2（默认0）
--image_border INT         # 计算指标时的边界裁剪（默认0）
--test_mask_csv PATH       # 盲元CSV路径或test_mask目录
--model NAME               # 模型名称（默认aacnet）
```

## 📂 数据集结构

```
/home/student_server/Qtt/NAFNet/data/
├── test_blur/              # 含盲元的破损图像
├── test_sharp/             # 原始清晰图像（参考）
├── test_mask/              # 盲元位置信息
│   └── <group_name>/
│       ├── blind_coords.csv         # 盲元坐标（静态）
│       └── flash_pixel_coords.csv   # 帧级坐标（动态）
└── ... (val_*, train_*)
```

## 📈 输出结果

```
./results/aacnet_blind_test_YYYYMMDD_HHMMSS/
├── test/                   # 补完后的图像
│   └── <group_name>/
│       ├── image_001.png
│       └── ...
└── blind_eval/             # 评估指标CSV
    ├── <group_name>/
    │   └── test_blind_metrics.csv   # 组级指标
    └── test_blind_metrics.csv       # 全局聚合指标
```

### 指标含义

| 指标 | 说明 | 范围 |
|------|------|------|
| PSNR | 全图峰值信噪比 | 越高越好（>30dB） |
| SSIM | 全图结构相似性 | 0~1，越高越好 |
| blind_mae | 盲元平均绝对误差 | 越小越好 |
| blind_rmse | 盲元均方根误差 | 越小越好 |
| blind_psnr | 盲元区域PSNR | 越高越好 |
| blind_mae_gain | MAE改进百分比 | 越高越好（%） |

## 🔧 技术细节

### 盲元掩码处理

脚本自动支持3种格式：

**1. 掩码图像** (`test_mask/<group>/image.png`)
```
灰度图，0=盲元，255=有效
```

**2. 静态坐标CSV** (`test_mask/<group>/blind_coords.csv`)
```csv
x,y
100,50
101,50
```

**3. 动态坐标CSV** (`test_mask/<group>/flash_pixel_coords.csv`)
```csv
frame_name,x,y
image_001.png,100,50
```

### 模型集成

```python
# Generator 输入
img: [B, 3, H, W]   范围 [-1, 1]
mask: [B, 1, H, W]  范围 [0, 1]（1=有效，0=盲元）

# 输出融合策略
output = generated × (1 - mask) + original × mask
```

## 🔍 常见问题

### Q: 如何处理没有测试集的情况？
A: 脚本需要 `test_blur` 和 `test_sharp` 目录。如果数据组织不同，需要调整 `--data_root` 指向正确的位置。

### Q: CSV坐标找不到警告
A: 检查：
- CSV文件位置：`test_mask/<group_name>/blind_coords.csv`
- CSV列名：必须包含 `x` 和 `y`（区分大小写）
- CSV编码：推荐 UTF-8 with BOM

### Q: 如何在多块GPU上加速？
A: 
```bash
python test_blind_aacnet.py \
    --data_root /path \
    --checkpoint /model.pth \
    --gpu_ids 0,1,2,3
```

### Q: 如何只评估特定的图像？
A: 修改 `test_blur` 目录，只保留目标图像文件。

### Q: 支持的checkpoint格式？
A: 支持以下3种：
```python
# 方式1：纯state_dict
torch.save(state_dict, 'model.pth')

# 方式2：包装格式
torch.save({'model': state_dict}, 'model.pth')

# 方式3：DataParallel（自动处理module前缀）
torch.save(model.state_dict(), 'model.pth')
```

## 📝 完整使用示例

### 场景1：基础测试
```bash
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint ./checkpoints/model_latest.pth
```

### 场景2：完整配置
```bash
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint ./checkpoints/model_latest.pth \
    --save_dir ./results/aacnet_v1 \
    --device cuda \
    --gpu_ids 0,1 \
    --image_border 0 \
    --test_mask_csv /home/student_server/Qtt/NAFNet/data/test_mask
```

### 场景3：使用启动工具
```bash
# 交互模式
python launch_blind_test.py

# 或直接指定参数
python launch_blind_test.py \
    --data_root /path \
    --checkpoint /path/model.pth \
    --non-interactive
```

## 📚 扩展和定制

### 添加新的评估指标

编辑 `test_blind_aacnet.py` 中的 `TestReport` 类：

```python
class TestReport:
    def update_metric(self, gt_img, out_img, img_name=None):
        # 添加自己的指标计算
        lpips = calculate_lpips(out_img, gt_img)
        self.total_lpips.append(lpips)
```

### 自定义预处理

修改 `to_tensor_rgb()` 和 `tensor_to_uint8_rgb()` 函数以支持不同的归一化方式。

## 🐛 故障排查

### 内存溢出
- 减少批处理大小（目前固定为1）
- 在CPU上运行：`--device cpu`
- 减小图像尺寸

### GPU显存不足
```bash
python test_blind_aacnet.py ... --device cpu
```

### 模型权重加载失败
```python
# 检查checkpoint格式
import torch
ckpt = torch.load('model.pth', map_location='cpu')
print(type(ckpt), ckpt.keys() if isinstance(ckpt, dict) else 'tensor')
```

## 📞 支持和文档

详细文档请见：[TEST_BLIND_GUIDE.md](TEST_BLIND_GUIDE.md)

## 📄 许可

遵循项目原有许可证。

---

## 版本信息

- **创建日期**: 2026-05-26
- **AACNet** 盲元补完测试框架
- **兼容性**: Python 3.7+, PyTorch 1.9+
