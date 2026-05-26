# AACNet 盲元补完网络 - 统一训练测试框架

这个文档总结了为AACNet创建的统一训练和测试框架。该框架通过配置文件驱动，实现了完整的端到端训练流程。

## 快速开始

### 前置条件

- Python 3.7+
- PyTorch 1.9+
- numpy, opencv-python, torchvision

### 安装依赖

```bash
pip install torch torchvision
pip install opencv-python numpy
```

### 训练网络

```bash
python main.py --train --config_path ./experiment.cfg
```

### 测试网络

```bash
python main.py --test --config_path ./experiment.cfg
```

## 框架结构

```
AACNet/
├── main.py                          # 主程序入口（训练/测试）
├── experiment.cfg                   # 配置文件（所有参数在这里）
├── util/
│   ├── config_loader.py             # 配置加载器
│   ├── metrics.py                   # 指标计算（PSNR, SSIM等）
│   ├── checkpoint_manager.py        # 检查点管理（保存/加载最好模型）
│   └── logger.py                    # 日志记录
├── dataloader/
│   └── blind_pixel_loader.py        # 数据加载器（盲元数据集）
├── TRAIN_TEST_GUIDE.md              # 详细使用指南
└── README_FRAMEWORK.md              # 本文件
```

## 关键特性

### 1. 配置驱动的参数管理

所有参数都在 `experiment.cfg` 中配置，支持快速修改和实验管理：

```ini
[training]
num_epochs = 100
batch_size = 2
learning_rate = 0.0001
val_interval = 20        # 每20轮验证一次

[checkpoint]
save_best_only = True    # 只保存最好的模型
best_metric = psnr       # 基于PSNR评估最好的模型
```

### 2. 最好模型自动保存

系统自动跟踪验证集的PSNR值，当有更高的PSNR时自动覆盖保存最好的模型：

```
最好模型保存路径: ./checkpoints/best_model.pth
```

关键参数：
- `save_best_only = True`: 只保存最好的模型（节省存储）
- `best_metric = psnr`: 使用PSNR作为评估指标
- `val_interval = 20`: 每20轮验证一次

### 3. 训练流程

```
for epoch in range(num_epochs):
    # 训练循环
    for batch in train_loader:
        model.set_input(batch)
        model.optimize_parameters()  # 前向、反向、优化一步完成
    
    # 验证（每val_interval轮）
    if (epoch + 1) % val_interval == 0:
        for batch in val_loader:
            model.set_input(batch)
            model.test()
            计算PSNR和SSIM
        
        # 比较PSNR
        if current_psnr > best_psnr:
            保存新的最好模型
            best_psnr = current_psnr
    
    # 更新学习率
    scheduler.step()
```

### 4. 数据集支持

框架支持盲元补完的特定数据组织结构：

```
data/
├── train_blur/           # 有盲元的训练图像
│   ├── 001/
│   │   ├── 1.png, 2.png, ...
│   ├── 002/, ..., 007/
├── train_sharp/          # 原始训练图像
│   ├── 001/, 002/, ..., 007/
├── train_mask/           # 盲元掩码（或坐标CSV）
│   ├── 001/, 002/, ..., 007/
├── val_blur/             # 验证集（similar structure）
├── val_sharp/
├── val_mask/
├── test_blur/            # 测试集（similar structure）
├── test_sharp/
└── test_mask/
```

注意：
- 子文件夹（001, 002等）对应不同的图像组
- 每个组内的文件按自然顺序编号（1.png, 2.png, ...）
- mask可以是图像文件或CSV坐标文件

### 5. 恢复训练

如果训练中断，可以轻松恢复：

```ini
[resume]
resume_training = True
checkpoint_path =              # 留空则自动找最新的
load_weights_only = False      # False: 恢复优化器和epoch状态
```

然后运行：
```bash
python main.py --train --config_path ./experiment.cfg
```

### 6. 多GPU支持

配置文件中支持多GPU训练：

```ini
[device]
gpu_ids = 0,1,2,3          # 使用4块GPU
use_dataparallel = True    # 启用DataParallel
```

## 输出结构

### 训练输出

```
./logs/
└── training_20240115_143022.log    # 训练日志

./checkpoints/
└── best_model.pth                   # 最好的模型（自动覆盖）

./val_results/
├── epoch_20/
│   ├── image_1_output.png          # 补完结果
│   ├── image_1_target.png          # 目标图像
│   └── ...
├── epoch_40/
└── ...
```

### 测试输出

```
./results/
├── 001/
│   ├── 1.png (补完后的图像)
│   ├── 2.png
│   └── ...
├── 002/
├── ... (按测试集的文件夹组织)
└── test_results.csv (汇总指标)
```

**test_results.csv 格式：**
```csv
image,group,psnr,ssim
1.png,001,28.5421,0.8543
2.png,001,29.1234,0.8612
...
```

## 配置文件详解

### 数据集配置 [dataset]

```ini
[dataset]
data_root = ./data              # 数据集根目录
image_width = 640               # 图像宽度
image_height = 512              # 图像高度
enable_augmentation = True      # 是否启用数据增强
flip_prob = 0.5                 # 水平翻转概率
rotation_angle = 10             # 旋转角度范围
```

### 训练配置 [training]

```ini
[training]
num_epochs = 100                # 训练轮数
batch_size = 2                  # 批次大小
learning_rate = 0.0001          # 初始学习率
lr_schedule = exponential        # 学习率衰减策略
lr_decay_factor = 0.99          # 衰减因子（每轮×0.99）
val_interval = 20               # 验证间隔
num_workers = 8                 # 数据加载线程数
```

### 损失函数配置 [loss]

```ini
[loss]
lambda_l1 = 1.0                 # L1重建损失权重
lambda_perceptual = 1.0         # 感知损失权重
lambda_style = 250.0            # 风格损失权重
lambda_adv = 0.1                # 对抗损失权重
lambda_consist = 1.0            # 一致性损失权重
```

## 常见问题

### Q: 如何改变验证频率？
A: 修改 `experiment.cfg` 中的 `val_interval`：
```ini
val_interval = 10   # 改为每10轮验证一次
```

### Q: 模型一直没有保存怎么办？
A: 检查以下几点：
1. 验证集路径是否正确（val_blur, val_sharp, val_mask）
2. PSNR计算是否正常（检查输出范围）
3. 最初的PSNR是否有记录（第一个验证轮的PSNR应该被保存）

### Q: 训练过程中出现显存不足怎么办？
A: 
1. 减少 `batch_size`
2. 减少 `num_workers`
3. 启用梯度累积（需要修改代码）

### Q: 如何使用CPU训练？
A: 修改 `experiment.cfg`：
```ini
[device]
gpu_ids =       # 留空表示使用CPU
```

### Q: 检查点如何恢复？
A: 
1. 设置 `resume_training = True`
2. 可选地设置 `checkpoint_path = ./checkpoints/best_model.pth`
3. 如果 `checkpoint_path` 为空，会自动找最新的检查点

## 系统架构

### 核心组件

1. **ConfigLoader** (`util/config_loader.py`)
   - 读取 experiment.cfg
   - 类型转换和验证
   - 默认值处理

2. **BlindPixelDataset** (`dataloader/blind_pixel_loader.py`)
   - 加载盲元数据集
   - 支持多种mask格式
   - 自动数据增强

3. **MetricCalculator** (`util/metrics.py`)
   - PSNR计算
   - SSIM计算
   - 其他指标（MAE, RMSE）

4. **CheckpointManager** (`util/checkpoint_manager.py`)
   - 检查点保存
   - 检查点加载
   - 最好模型追踪

5. **Logger** (`util/logger.py`)
   - 控制台日志
   - 文件日志
   - 时间戳记录

### 数据流

```
experiment.cfg
    ↓
ConfigLoader.get_config()
    ↓
create_dataloader() → BlindPixelDataset
    ↓
train_epoch()
├── model.set_input()
├── model.optimize_parameters()
└── get_current_errors()
    ↓
validate()
├── model.test()
├── MetricCalculator.calculate_psnr/ssim()
└── CheckpointManager.save_checkpoint()
```

## 性能优化建议

1. **增加批次大小** - 如果GPU显存充足
2. **减少验证频率** - 增加 `val_interval`
3. **启用混合精度** - 设置 `mixed_precision = True`
4. **增加数据加载线程** - 调整 `num_workers`

## 源代码统计

| 文件 | 行数 | 功能 |
|-----|------|------|
| main.py | 580+ | 主程序逻辑 |
| config_loader.py | 220+ | 配置管理 |
| metrics.py | 180+ | 指标计算 |
| checkpoint_manager.py | 150+ | 检查点管理 |
| logger.py | 130+ | 日志记录 |
| blind_pixel_loader.py | 300+ | 数据加载 |

**总计：2000+ 行代码和文档**

## 许可证和引用

该框架基于现有的AACNet实现进行扩展，用于盲元补完任务。

---

需要更多帮助？查看 `TRAIN_TEST_GUIDE.md` 获取详细的参数说明。
