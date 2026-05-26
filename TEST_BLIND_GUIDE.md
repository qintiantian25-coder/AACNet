# AACNet 盲元数据集测试指南

## 概述
本指南详细说明如何使用AACNet在盲元（dead pixels/bad pixels）补完数据集上进行测试。

## 数据集结构要求

```
/home/student_server/Qtt/NAFNet/data/
├── test_blur/          # 测试集含盲元的图像（RGB）
├── test_sharp/         # 测试集原始清晰图像（RGB）
├── test_mask/          # 测试集盲元位置掩码或CSV坐标文件
├── val_blur/           # 验证集含盲元的图像
├── val_sharp/          # 验证集原始清晰图像
├── val_mask/           # 验证集盲元掩码
├── train_blur/         # 训练集含盲元的图像
├── train_sharp/        # 训练集原始清晰图像
└── train_mask/         # 训练集盲元掩码或CSV文件
```

### 盲元掩码格式

#### 选项 1：Mask 图像文件
- 文件名：与对应的输入图像相同（如 `image.png` 对应 `image.png`）
- 格式：灰度或二值图，0 表示盲元，255 表示有效区域
- 位置：`test_mask/<group_name>/image.png`

#### 选项 2：CSV 坐标文件
- 文件名：`blind_coords.csv` 或 `blind_pixel_coords.csv`
- 格式：
  ```
  x,y
  100,50
  101,50
  102,50
  ...
  ```
- 位置：`test_mask/<group_name>/blind_coords.csv`

#### 选项 3：闪光盲元 CSV（帧级动态坐标）
- 文件名：`flash_pixel_coords.csv`
- 格式：
  ```
  frame_name,x,y
  image_001.png,100,50
  image_001.png,101,50
  image_002.png,102,51
  ...
  ```
- 位置：`test_mask/<group_name>/flash_pixel_coords.csv`

## 安装依赖

确保已安装所需的Python包：

```bash
cd /home/tianyu/Pythonproject/AACNet
pip install -r requirements.txt
```

## 运行测试

### 基本用法

```bash
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint /path/to/your/model_checkpoint.pth \
    --save_dir ./results/aacnet_blind_test \
    --device cuda \
    --model aacnet \
    --name aacnet_blind
```

### 参数说明

| 参数 | 类型 | 必需 | 默认值 | 说明 |
|------|------|------|--------|------|
| `--data_root` | str | 是 | - | 数据集根目录 |
| `--checkpoint` | str | 是 | - | 预训练模型路径 (.pth) |
| `--save_dir` | str | 否 | `results/aacnet_blind_test` | 结果保存目录 |
| `--device` | str | 否 | `cuda` | 计算设备 (cuda/cpu) |
| `--test_mask_csv` | str | 否 | None | 盲元CSV路径或test_mask目录 |
| `--image_border` | int | 否 | 0 | PSNR/SSIM计算时的裁剪边界像素 |
| `--model` | str | 否 | `aacnet` | 模型名称 |
| `--name` | str | 否 | `aacnet_blind` | 模型实例名称 |
| `--checkpoints_dir` | str | 否 | `./checkpoints` | 检查点目录 |
| `--gpu_ids` | str | 否 | `0` | GPU IDs，多GPU用逗号分隔 |

### 完整示例

```bash
# 测试单个模型
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint ./checkpoints/aacnet_blind/net_G_latest.pth \
    --save_dir ./results/aacnet_blind_test_v1 \
    --device cuda \
    --image_border 0 \
    --model aacnet \
    --gpu_ids 0

# 使用指定的盲元掩码目录
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint ./checkpoints/aacnet_blind/net_G_latest.pth \
    --save_dir ./results/aacnet_blind_test \
    --test_mask_csv /home/student_server/Qtt/NAFNet/data/test_mask \
    --device cuda

# 多GPU测试
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint ./checkpoints/aacnet_blind/net_G_latest.pth \
    --gpu_ids 0,1,2,3 \
    --save_dir ./results/aacnet_blind_test
```

## 输出结果

测试完成后，结果保存在 `--save_dir` 目录中：

```
results/aacnet_blind_test/
├── test/                    # 补完结果图像
│   └── <group_name>/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
└── blind_eval/              # 评估指标
    ├── <group_name>/
    │   └── test_blind_metrics.csv   # 组级指标
    └── test_blind_metrics.csv       # 全局指标
```

### CSV 格式说明

#### Per-Image 指标 (test_blind_metrics.csv)

| 字段 | 说明 |
|------|------|
| `image` | 图像文件名 |
| `psnr` | 全图 PSNR (dB) |
| `ssim` | 全图 SSIM |
| `blind_mae` | 盲元区域平均绝对误差 |
| `blind_rmse` | 盲元区域均方根误差 |
| `blind_psnr` | 盲元区域 PSNR (dB) |
| `blind_mae_input` | 输入图像在盲元区域的 MAE |
| `blind_mae_gain_abs` | MAE 绝对改进 |
| `blind_mae_gain_pct` | MAE 百分比改进 (%) |
| `blind_count` | 评估的盲元像素总数 |

## 评估指标说明

### 全图指标
- **PSNR**：峰值信号噪声比（越高越好，一般>30dB）
- **SSIM**：结构相似性指数（-1到1，越接近1越好）

### 盲元区域指标
- **Blind MAE**：盲元像素上的平均绝对误差（越小越好）
- **Blind RMSE**：盲元像素上的均方根误差（越小越好）
- **Blind PSNR**：基于盲元像素的PSNR（越高越好）
- **MAE Gain**：相对于输入图像的改进程度

## 常见问题

### Q: 如何处理没有盲元坐标的情况？
A: 脚本会自动创建全1的mask（表示全部有效），此时盲元指标将为空。确保提供正确的盲元坐标CSV文件。

### Q: CSV文件找不到的警告
A: 检查以下几点：
1. CSV文件是否存在于 `test_mask/<group_name>/` 目录
2. CSV文件名是否正确（`blind_coords.csv` 或 `blind_pixel_coords.csv`）
3. CSV文件是否有正确的表头（必须包含 `x` 和 `y` 列）

### Q: 如何在多个GPU上并行测试？
A: 使用 `--gpu_ids` 参数指定多个GPU，但注意当前实现每次处理一张图像（batchSize=1），多GPU主要用于加速单张图像的处理。

### Q: 如何只处理特定的图像子集？
A: 修改 `test_blur` 目录，只保留需要测试的图像文件。脚本会自动扫描该目录的所有PNG文件。

## 模型检查点格式

支持以下checkpoint格式：
```python
# 格式 1：纯 state_dict
state_dict = {...}
torch.save(state_dict, 'net_G_latest.pth')

# 格式 2：包装的 state_dict
checkpoint = {'model': state_dict, ...}
torch.save(checkpoint, 'net_G_latest.pth')

# 格式 3：DataParallel 格式（自动处理 'module.' 前缀）
checkpoint = {
    'module.layer1.weight': ...,
    ...
}
```

## 脚本特性

### 自动适配
- ✅ 自动处理RGB图像（从BGR转换）
- ✅ 自动调整输出图像尺寸以匹配GT
- ✅ 支持多种盲元坐标格式
- ✅ 自动按组织结构组织结果
- ✅ 支持缺失的GT图像

### 输出数据
- ✅ 补完结果图像（RGB格式）
- ✅ 逐图像指标CSV
- ✅ 全局聚合指标
- ✅ 盲元区域详细分析

## 扩展使用

### 定制指标计算

如需添加其他指标（如LPIPS、FID等），可以：
1. 在 `TestReport` 类中添加新的计算方法
2. 在 `update_metric()` 方法中调用
3. 在输出CSV中添加新字段

### 修改预处理/后处理

编辑 `to_tensor_rgb()` 和 `tensor_to_uint8_rgb()` 函数来自定义图像处理方式。

## 参考论文和资源

- AACNet: Aggregated Attention Convolution Networks  
- 盲元补完相关工作参考用户提供的NAFNet实现

## 联系方式

如有问题，请检查：
1. 数据集路径和格式
2. 模型检查点是否有效
3. GPU显存是否充足（建议至少8GB）
