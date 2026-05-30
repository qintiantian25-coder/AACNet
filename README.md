# AACNet 红外图像盲元修复说明文档

本仓库用于红外灰度图像的盲元修复任务，核心目标是从带盲元退化的输入图像中恢复清晰结果，并在验证与测试阶段输出全图指标与盲元专项指标。

本版本已经整理为单卡运行流程，默认入口如下：

```bash
python main.py --train --config_path ./experiment.cfg
python main.py --test --config_path ./experiment.cfg
```

如果你只想测试，也可以直接运行：

```bash
python ./test.py
```

## 1. 任务目标

AACNet 面向的是红外图像中的盲元修复。数据集中的图像均为灰度图像，输入图像包含传感器盲元退化，模型的任务是结合盲元坐标信息、闪元信息和图像上下文，重建出更接近清晰真值的结果。

本项目在训练、验证、测试阶段的职责划分如下：

- 训练阶段：仅使用 `blind_pixel_coords.csv` 构造训练 mask，让模型学习修复静态盲元区域。
- 验证阶段：除全图 PSNR/SSIM 外，同时叠加 `flash_pixel_coords.csv`，计算 blind MAE / RMSE / PSNR 等盲元专项指标。
- 测试阶段：与验证阶段保持一致，同样叠加 `flash_pixel_coords.csv` 进行 blind 指标统计。

## 2. 数据集目录结构

配置文件默认使用项目内相对路径 `./data_new`。数据集按训练、验证、测试三部分划分，每部分内部再按数据组组织，例如 `001`、`002`、`003` 等。

推荐目录结构如下：

```text
data_new/
├── train_blur/
│   ├── 001/
│   ├── 002/
│   └── ...
├── train_sharp/
│   ├── 001/
│   ├── 002/
│   └── ...
├── train_mask/
│   ├── 001/
│   │   ├── blind_pixel_coords.csv
│   │   ├── blind_pixel_mask.png
│   │   └── flash_pixel_coords.csv
│   ├── 002/
│   └── ...
├── val_blur/
├── val_sharp/
├── val_mask/
└── test_blur/
    test_sharp/
    test_mask/
```

### 2.1 各目录含义

- `*_blur/`：退化输入图像，包含盲元噪声。
- `*_sharp/`：与输入严格一一对应的清晰真值图像。
- `*_mask/`：盲元标注信息目录。

### 2.2 每个组目录中的文件说明

每个组目录，例如 `train_mask/001/`，通常包含以下文件：

- `blind_pixel_coords.csv`
  - 静态盲元坐标文件，优先作为训练时 mask 的来源。
  - CSV 中常见字段为 `x`、`y`，表示盲元坐标。
- `blind_pixel_mask.png`
  - 静态盲元掩码图像，作为坐标文件的兼容回退。
  - 白色像素通常表示盲元，加载时会反转成模型所需语义：有效区域为 1，盲元区域为 0。
- `flash_pixel_coords.csv`
  - 帧级闪元坐标文件。
  - 仅用于验证与测试阶段的盲元专项指标统计，不参与训练输入 mask 的构造。

## 3. 本项目实际采用哪些文件

这部分是重点，按阶段说明如下。

### 3.1 训练阶段使用的文件

训练入口是 `main.py --train`，训练过程中主要读取以下文件：

- `experiment.cfg`
  - 读取数据集路径、图像尺寸、batch size、学习率、checkpoint 路径等配置。
- `dataloader/blind_pixel_loader.py`
  - 负责加载 `train_blur`、`train_sharp`、`train_mask`。
  - 训练时 mask 读取优先级为：`blind_pixel_coords.csv` -> `blind_pixel_mask.png`。
  - `flash_pixel_coords.csv` 不参与训练输入 mask。
- `model/aacnet_model.py`
  - AACNet 主模型封装。
  - 接收图像与 mask，输出修复结果。
- `model/aacnet.py`、`model/network.py`、`model/base_function.py`
  - 模型主体结构与 mask 引导的网络实现。
- `util/checkpoint_manager.py`
  - 管理模型保存与恢复。
  - 训练中非最佳状态写入 `last_state.pt`，验证指标提升时写入 `best_model.pt`。
- `util/logger.py`
  - 记录训练日志与验证日志。

### 3.2 验证阶段使用的文件

验证也通过 `main.py` 的训练流程在每个 `val_interval` 周期触发，验证阶段读取：

- `val_blur/`
- `val_sharp/`
- `val_mask/`

验证时的行为：

- 训练输入仍然只依赖 `blind_pixel_coords.csv` 构造 mask。
- 计算验证指标时，会同时使用：
  - `blind_pixel_coords.csv`
  - `flash_pixel_coords.csv`
- 盲元专项指标包括：
  - `blind_mae`
  - `blind_rmse`
  - `blind_psnr`
- 全图指标包括：
  - `psnr`
  - `ssim`

### 3.3 测试阶段使用的文件

测试入口是 `main.py --test`，它会复用 `test.py` 中的单卡测试函数。测试时读取：

- `test_blur/`
- `test_sharp/`
- `test_mask/`

测试阶段同样遵循以下规则：

- 模型权重只加载 `./experiments/models/best_model.pt`。
- 全图指标仍然是 `psnr`、`ssim`。
- 盲元专项指标同验证阶段一致：`blind_mae`、`blind_rmse`、`blind_psnr`。
- `flash_pixel_coords.csv` 只用于测试盲元统计，不参与输入 mask 构造。

## 4. 模型文件说明

本项目的检查点目录默认是：

```text
./experiments/models/
```

当前约定的模型文件如下：

- `best_model.pt`
  - 最佳模型文件。
  - 只有当验证指标提升时才会更新。
  - 测试阶段默认加载这个文件。
- `last_state.pt`
  - 最新训练状态文件。
  - 每次训练保存都会更新，方便中断后继续训练。

如果目录中存在旧版本文件，当前逻辑会尽量清理掉，只保留以上两个核心文件。

## 5. 训练流程说明

训练流程入口：

```bash
python main.py --train --config_path ./experiment.cfg
```

训练步骤如下：

1. 读取 `experiment.cfg`。
2. 加载 `train_blur`、`train_sharp`、`train_mask`。
3. 对每个组目录优先读取 `blind_pixel_coords.csv`，如不存在则回退到 `blind_pixel_mask.png`。
4. 只用静态盲元坐标构造训练输入 mask。
5. 将输入图像和 mask 送入 AACNet。
6. 使用 L1 损失进行训练。
7. 每个 epoch 结束后保存 `last_state.pt`。
8. 每次验证时计算 `psnr` 和 `ssim`，并同时计算 blind 专项指标。
9. 如果验证指标优于历史最佳，则更新 `best_model.pt`。

### 5.1 训练输出

训练完成后，主要输出文件为：

```text
./experiments/models/best_model.pt
./experiments/models/last_state.pt
./experiments/logs/training.txt
./experiments/logs/validation.txt
```

### 5.2 是否可以断点续训

可以。当前逻辑会优先使用 `last_state.pt` 做恢复训练，其次才会回退到 `best_model.pt`。

## 6. 验证流程说明

验证在训练过程中自动执行，无需单独启动。

验证阶段会：

1. 读取 `val_blur`、`val_sharp`、`val_mask`。
2. 对每个组优先读取 `blind_pixel_coords.csv`。
3. 将 `flash_pixel_coords.csv` 叠加进 blind 指标统计。
4. 计算全图 `psnr`、`ssim`。
5. 计算 blind 专项 `blind_mae`、`blind_rmse`、`blind_psnr`。
6. 若当前验证指标刷新最佳，则更新 `best_model.pt`。

验证日志会输出两类信息：

- 当前验证的 `psnr`、`ssim`、`loss`
- 是否刷新最佳模型的提示

## 7. 测试流程说明

测试入口：

```bash
python main.py --test --config_path ./experiment.cfg
```

测试阶段流程如下：

1. 读取 `test_blur`、`test_sharp`、`test_mask`。
2. 自动加载 `./experiments/models/best_model.pt`。
3. 将测试输出按组写入 `./results/aacnet_blind_test/test/<group>/`。
4. 同时生成三联图到 `./results/aacnet_blind_test/triple_comparison/<group>/`，三联图内容为：`输入中心帧 | 模型修复结果 | 真值GT`。
5. 对每个组优先读取 `blind_pixel_coords.csv`，再结合 `flash_pixel_coords.csv` 做 blind 统计。
6. 计算全图 `psnr`、`ssim`。
7. 计算 blind 专项 `blind_mae`、`blind_rmse`、`blind_psnr`。
8. 每个组推理完成后，立即生成该组的 CSV：`./results/aacnet_blind_test/blind_eval/<group>/test_blind_metrics.csv`。
9. 全部测试完成后，再汇总生成全量 CSV：`./results/aacnet_blind_test/blind_eval/test_blind_metrics.csv`。

### 7.1 测试输出

测试结果通常保存到：

```text
./results/aacnet_blind_test/
```

其中一般会包含：

- `triple_comparison/`：三联图，按组保存。
- `test/`：修复后的输出图像，按组保存。
- `blind_eval/<group>/test_blind_metrics.csv`：各组盲元指标 CSV。
- `blind_eval/test_blind_metrics.csv`：全测试集汇总 CSV。

## 8. 推荐的配置文件

当前建议使用的配置项要点如下：

- 数据路径：`./data_new`
- 检查点目录：`./experiments/models`
- 最佳模型文件名：`best_model`
- GPU：单卡 `gpu_ids = 0`
- 数据增强：仅训练集开启
- 训练输入 mask：只读取 `blind_pixel_coords.csv`
- 验证/测试 blind 指标：`blind_pixel_coords.csv` + `flash_pixel_coords.csv`

对应配置文件是：

- [`experiment.cfg`](experiment.cfg)

## 9. 代码文件说明

下面是你在阅读或修改时最该关注的文件：

- [`main.py`](main.py)
  - 训练、验证、测试主入口。
- [`test.py`](test.py)
  - 单卡测试与 blind 指标统计实现。
- [`dataloader/blind_pixel_loader.py`](dataloader/blind_pixel_loader.py)
  - 数据读取、mask 构造、组级目录解析。
- [`model/aacnet_model.py`](model/aacnet_model.py)
  - 模型封装与前向推理。
- [`model/aacnet.py`](model/aacnet.py)
  - AACNet 主干结构。
- [`util/checkpoint_manager.py`](util/checkpoint_manager.py)
  - `best_model.pt` 与 `last_state.pt` 的保存、恢复。
- [`util/metrics.py`](util/metrics.py)
  - 全图 PSNR / SSIM 计算。

## 10. 常见问题

### 10.1 为什么训练不直接用 flash_pixel_coords.csv

因为 `flash_pixel_coords.csv` 代表的是帧级动态闪元，它更适合做验证/测试统计，而不是训练输入 mask 的主来源。训练中只用 `blind_pixel_coords.csv`，可以让模型先稳定学习静态盲元修复。

### 10.2 为什么验证和测试都要叠加 flash_pixel_coords.csv

因为验证/测试的 blind 指标希望更全面地反映模型在静态盲元和动态闪元上的恢复能力，所以把两类坐标合并统计更合理。

### 10.3 测试时加载的是哪个模型文件

默认加载：

```text
./experiments/models/best_model.pt
```

这是当前验证指标最好的模型权重文件。

### 10.4 训练中断后从哪里恢复

优先从：

```text
./experiments/models/last_state.pt
```

恢复最新训练状态。

## 11. 简短结论

如果你只记住一件事，那就是：

- 训练看 `blind_pixel_coords.csv`
- 验证/测试在 blind 指标里同时看 `blind_pixel_coords.csv + flash_pixel_coords.csv`
- 最佳模型文件是 `./experiments/models/best_model.pt`
- 最新训练状态文件是 `./experiments/models/last_state.pt`
