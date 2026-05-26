# AACNet 盲元补完网络 - 使用指南

## 快速开始

### 1. 训练

```bash
python main.py --train --config_path ./experiment.cfg
```

### 2. 测试

```bash
python main.py --test --config_path ./experiment.cfg
```

## 配置文件说明

所有参数都在 `experiment.cfg` 中配置。主要参数说明：

### 数据集配置 [dataset]

```ini
data_root = ./data                    # 数据集根目录
train_blur_dir = train_blur           # 训练集带盲元的图像目录名
train_sharp_dir = train_sharp         # 训练集清晰图像目录名
train_mask_dir = train_mask           # 训练集mask目录名
val_blur_dir = val_blur               # 验证集
val_sharp_dir = val_sharp
val_mask_dir = val_mask
test_blur_dir = test_blur             # 测试集
test_sharp_dir = test_sharp
test_mask_dir = test_mask

image_width = 640                     # 图像宽度（固定）
image_height = 512                    # 图像高度（固定）

enable_augmentation = True            # 是否启用数据增强（仅训练）
flip_prob = 0.5                      # 水平翻转概率
rotation_angle = 10                  # 旋转角度范围
```

### 训练配置 [training]

```ini
num_epochs = 100                      # 训练总轮数
batch_size = 2                        # 批次大小
learning_rate = 0.0001                # 初始学习率
lr_schedule = exponential             # 学习率衰减策略: exponential
lr_decay_factor = 0.99                # 学习率衰减系数（每轮×0.99）
val_interval = 20                     # 验证间隔（每20轮验证一次）
num_workers = 8                       # 数据加载线程数
shuffle = True                        # 是否洗牌训练数据
```

### 检查点配置 [checkpoint]

```ini
checkpoint_dir = ./checkpoints        # 检查点保存目录
model_prefix = best_model             # 最好模型的文件名前缀
save_interval = 0                     # 临时检查点保存间隔（0表示不保存）
save_best_only = True                 # 只保存最好的模型
best_metric = psnr                    # 评估指标: psnr 或 loss
```

### 其他重要参数

**优化器 [optimizer]**
```ini
optimizer_type = adam                 # 优化器类型: adam 或 sgd
adam_beta1 = 0.5
adam_beta2 = 0.9
```

**损失函数 [loss]**
```ini
lambda_l1 = 1.0                       # L1重建损失权重
lambda_perceptual = 1.0               # 感知损失权重
lambda_style = 250.0                  # 风格损失权重
lambda_adv = 0.1                      # 对抗损失权重
lambda_consist = 1.0                  # 一致性损失权重
```

**设备 [device]**
```ini
gpu_ids = 0                           # GPU ID，多个GPU用逗号分隔: 0,1,2,3
use_dataparallel = False              # 是否使用DataParallel
mixed_precision = False               # 混合精度训练
```

**恢复训练 [resume]**
```ini
resume_training = False               # 是否从检查点恢复
checkpoint_path =                     # 检查点路径（空则自动寻找最新）
load_weights_only = False             # 是否只加载权重
```

## 数据集结构

```
data/
├── train_blur/
│   ├── 001/
│   │   ├── 1.png
│   │   ├── 2.png
│   │   └── ...
│   ├── 002/
│   └── ... (001-007)
├── train_sharp/
│   ├── 001/
│   ├── 002/
│   └── ... (001-007)
├── train_mask/
│   ├── 001/
│   │   ├── blind_coords.csv         # 盲元坐标文件
│   │   └── 1.png                    # mask图像（可选）
│   └── ... (001-007)
├── val_blur/
│   ├── 001/
│   ├── 002/
│   └── ... (001-002)
├── val_sharp/
│   └── ... (001-002)
├── val_mask/
│   └── ... (001-002)
├── test_blur/
│   ├── 001/
│   ├── 002/
│   └── ... (001-006)
├── test_sharp/
│   └── ... (001-006)
└── test_mask/
    └── ... (001-006)
```

**重要**: 每个子文件夹内的文件名应该按照自然顺序编号（1.png, 2.png, ...）。

## 训练流程

1. **数据加载**: 从配置的目录中加载训练/验证/测试数据
2. **模型初始化**: 根据配置创建模型和优化器
3. **训练循环**:
   - 每个epoch遍历整个训练数据集
   - 计算损失并反向传播
   - 每`val_interval`轮进行一次验证
   - 基于验证PSNR判断是否为最好的模型
   - 如果PSNR更高，自动覆盖保存最好的模型
4. **学习率衰减**: 每轮后学习率乘以`lr_decay_factor`
5. **模型保存**: 仅保留一个最好的模型（best_model.pth）

## 模型保存和恢复

### 模型保存位置
- 最好的模型: `./checkpoints/best_model.pth`
- 临时检查点: `./checkpoints/best_model_epoch_X.pth` (如果`save_interval > 0`)

### 恢复训练
要继续之前的训练，修改`experiment.cfg`:
```ini
[resume]
resume_training = True
checkpoint_path =            # 留空则自动找最新的
load_weights_only = False
```

然后运行:
```bash
python main.py --train --config_path ./experiment.cfg
```

### 加载预训练模型进行测试
检查点会自动从 `./checkpoints/best_model.pth` 加载。

## 输出文件

### 训练输出

```
./logs/
└── training_YYYYMMDD_HHMMSS.log     # 训练日志文件

./checkpoints/
└── best_model.pth                    # 最好的模型

./val_results/
├── epoch_1/
│   ├── image_1_output.png
│   ├── image_1_target.png
│   └── ...
└── ...                               # 验证结果可视化
```

### 测试输出

```
./results/
├── 001/
│   ├── 1.png                         # 补完后的图像
│   ├── 2.png
│   └── ...
├── 002/
├── ... (001-006)
└── test_results.csv                  # CSV格式的指标
```

**test_results.csv 格式:**
```csv
image,group,psnr,ssim
1.png,001,28.5421,0.8543
2.png,001,29.1234,0.8612
...
```

## 常见问题

### Q: 如何修改验证间隔？
A: 在 `experiment.cfg` 的 `[training]` 部分修改 `val_interval`。例如：
```ini
val_interval = 10   # 每10轮验证一次
```

### Q: 如何使用多GPU？
A: 在 `experiment.cfg` 的 `[device]` 部分配置：
```ini
gpu_ids = 0,1,2,3               # 4块GPU
use_dataparallel = True         # 启用DataParallel
```

### Q: 如何改变批次大小？
A: 在 `experiment.cfg` 的 `[training]` 部分修改：
```ini
batch_size = 4    # 改为4
```

### Q: 如何查看训练日志？
A: 训练日志保存在 `./logs/` 目录下，以时间戳命名。

### Q: 如果训练中断了怎么办？
A: 设置 `resume_training = True` 然后重新运行训练命令，会自动从最新的检查点恢复。

### Q: 如何只使用模型权重而不恢复优化器状态？
A: 设置 `load_weights_only = True`

### Q: 最好的模型是如何定义的？
A: 默认以验证集的PSNR为标准，PSNR越高越好。可以改为其他指标：
```ini
best_metric = ssim   # 使用SSIM（越高越好）
best_metric = loss   # 使用损失（越低越好）
```

## 性能优化建议

1. **增加批次大小**: 如果GPU显存充足，增加 `batch_size` 可以加速训练
2. **减少数据加载线程**: 如果数据在本地SSD，可以减少 `num_workers`
3. **启用混合精度**: 设置 `mixed_precision = True` 可以加快训练（需要GPU支持）
4. **更改验证间隔**: 验证很耗时，可以设置更大的 `val_interval`

## 故障排除

### 错误: "配置文件不存在"
- 检查 `experiment.cfg` 是否在当前目录
- 使用正确的路径: `--config_path /path/to/experiment.cfg`

### 错误: "数据集目录不存在"
- 检查 `data_root` 配置是否正确
- 验证目录结构是否按照要求组织

### 错误: "CUDA out of memory"
- 减少 `batch_size`
- 使用 `--device cpu` 在CPU上运行（很慢）
- 启用梯度积累（需要修改代码）

### 训练非常慢
- 检查 `num_workers` 是否设置过高
- 尝试减少 `val_interval` 来加快training epoch的速度
- 确保数据在快速存储（SSD/NVMe）上

## 高级用法

### 自定义配置
可以创建多个配置文件，例如：
- `experiment.cfg` - 主配置
- `experiment_debug.cfg` - 调试配置（小数据集）
- `experiment_large.cfg` - 大规模训练配置

使用：
```bash
python main.py --train --config_path ./experiment_large.cfg
```

### 修改损失函数权重
在 `experiment.cfg` 的 `[loss]` 部分调整各个损失的权重来平衡不同的目标。

---

需要更多帮助？检查 `./logs/` 目录下的训练日志文件。
