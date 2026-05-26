# AACNet 盲元补完测试框架 - 文件索引

创建时间：2026-05-26

## 🎯 快速导航

### 我想要...
- **立即开始测试** → 运行 `python launch_blind_test.py` ✓
- **了解详细步骤** → 阅读 [TEST_BLIND_GUIDE.md](TEST_BLIND_GUIDE.md) ✓
- **查看快速参考** → 阅读 [BLIND_TEST_README.md](BLIND_TEST_README.md) ✓
- **修改配置参数** → 编辑 [config_blind_test.template](config_blind_test.template) ✓
- **用Bash脚本** → 编辑并运行 [run_blind_test.sh](run_blind_test.sh) ✓

---

## 📦 创建的文件详细说明

### 1️⃣ 核心测试脚本

#### `test_blind_aacnet.py` 【主测试脚本】
**位置**: `/home/tianyu/Pythonproject/AACNet/test_blind_aacnet.py`

**用途**: 
- AACNet在盲元数据集上的完整测试脚本
- 支持多种盲元坐标格式和掩码类型
- 计算PSNR、SSIM、blind_mae、blind_rmse、blind_psnr等指标

**关键功能**:
```python
- 盲元坐标加载：CSV、mask图像、闪光坐标
- 评估指标：全图指标 + 盲元区域特定指标
- 结果管理：自动按组织结构保存图像和CSV报告
- 兼容性：支持DataParallel、多种checkpoint格式
```

**使用方式**:
```bash
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint /path/to/model.pth \
    --save_dir ./results
```

**输入要求**:
- test_blur/ - RGB图像（含盲元）
- test_sharp/ - RGB图像（参考）
- test_mask/ - 盲元坐标或mask

**输出内容**:
- results/test/\<group\>/ - 补完结果（RGB）
- results/blind_eval/ - CSV格式评估指标

---

### 2️⃣ 模型定义文件

#### `model/aacnet_model.py` 【模型封装】
**位置**: `/home/tianyu/Pythonproject/AACNet/model/aacnet_model.py`

**用途**: 
- 为测试框架定义AACNetBlind模型类
- 继承BaseModel，兼容现有的模型加载机制

**关键代码**:
```python
class AACNetBlind(BaseModel):
    def set_input(self, input_data, epoch=0):
        # 处理输入：img [B,3,H,W] + mask [B,3,H,W]
        
    def test(self):
        # 模型推理并融合结果
        # img_out = generated * (1-mask) + original * mask
```

**配置**:
- ngf=48 - 生成器基础通道数
- 继承自BaseModel - 自动处理GPU分配

---

### 3️⃣ 启动和配置工具

#### `launch_blind_test.py` 【交互式启动】
**位置**: `/home/tianyu/Pythonproject/AACNet/launch_blind_test.py`

**用途**: 
- 提供交互式和命令行的启动方式
- 参数验证和提示

**使用方式**:
```bash
# 交互模式（推荐新手）
python launch_blind_test.py

# 命令行模式
python launch_blind_test.py \
    --data_root /path \
    --checkpoint /path/model.pth \
    --non-interactive
```

**特点**:
- ✓ 交互式参数输入和验证
- ✓ 自动检查目录存在性
- ✓ 测试完成后显示结果摘要

---

#### `run_blind_test.sh` 【Shell脚本】
**位置**: `/home/tianyu/Pythonproject/AACNet/run_blind_test.sh`

**用途**: 
- Bash环境下的快速启动脚本
- 方便在HPC集群或shell脚本中使用

**使用方式**:
```bash
# 编辑脚本顶部的参数配置
vim run_blind_test.sh

# 运行
bash run_blind_test.sh
```

**配置参数**（脚本顶部）:
```bash
DATA_ROOT="/home/student_server/Qtt/NAFNet/data"
CHECKPOINT="/path/to/model.pth"
SAVE_DIR="./results/test_$(date +%Y%m%d_%H%M%S)"
GPU_IDS="0"
```

---

#### `config_blind_test.template` 【配置模板】
**位置**: `/home/tianyu/Pythonproject/AACNet/config_blind_test.template`

**用途**: 
- 提供所有可用参数的详细说明
- 可复制并修改后直接使用

**使用方式**:
```bash
# 复制模板
cp config_blind_test.template config_blind_test.txt

# 编辑参数
vim config_blind_test.txt

# 使用参数文件运行
python test_blind_aacnet.py $(cat config_blind_test.txt)
```

**包含内容**:
- ✓ 所有参数详细说明
- ✓ 数据集结构要求
- ✓ CSV格式说明
- ✓ 评估指标解释

---

### 4️⃣ 文档和指南

#### `TEST_BLIND_GUIDE.md` 【完整指南】
**位置**: `/home/tianyu/Pythonproject/AACNet/TEST_BLIND_GUIDE.md`

**内容**:
- 📋 概述和功能介绍
- 📂 数据集结构详细说明
- 🔧 安装依赖和运行方式
- 📊 参数详细说明表
- 📈 输出结果格式说明
- ❓ 常见问题解答
- 🔌 扩展和定制方法

**适合**: 需要全面了解框架的用户

**关键章节**:
- 数据集结构和盲元掩码格式
- 完整参数说明表
- CSV格式规范
- 评估指标详解
- 常见问题

---

#### `BLIND_TEST_README.md` 【快速参考】
**位置**: `/home/tianyu/Pythonproject/AACNet/BLIND_TEST_README.md`

**内容**:
- 📋 文件清单速查表
- 🚀 三种快速开始方式
- 📊 核心参数速查
- 📂 数据集结构（简化版）
- 📈 输出结果概览
- 🔧 技术细节
- 🔍 常见问题（精简版）
- 📝 完整使用示例

**适合**: 熟悉AACNet、需要快速上手的用户

---

#### `INDEX.md` 【本文件】
**位置**: `/home/tianyu/Pythonproject/AACNet/INDEX.md`

**用途**: 
- 总览所有创建的文件
- 提供快速导航
- 说明各文件的用途和使用方法

---

## 📊 文件关系图

```
用户运行脚本
      ↓
launch_blind_test.py  ← 交互式启动（推荐）
      ↓
test_blind_aacnet.py  ← 主测试脚本
      ├─→ model/aacnet_model.py  ← 模型定义
      ├─→ 加载数据集
      ├─→ 推理 + 评估
      └─→ 输出结果
      
支持配置方式：
- 命令行参数
- config_blind_test.template（参数文件）
- run_blind_test.sh（Shell脚本）

文档支持：
- BLIND_TEST_README.md（快速参考）
- TEST_BLIND_GUIDE.md（详细指南）
- config_blind_test.template（参数说明）
```

---

## 🎓 学习路径

### 初级用户
1. 阅读 [BLIND_TEST_README.md](BLIND_TEST_README.md) - 了解基本概念
2. 运行 `python launch_blind_test.py` - 交互式体验
3. 查看输出结果结构

### 中级用户
1. 阅读 [TEST_BLIND_GUIDE.md](TEST_BLIND_GUIDE.md) - 深入了解
2. 修改 [config_blind_test.template](config_blind_test.template) - 自定义参数
3. 运行 `python test_blind_aacnet.py $(cat config_blind_test.txt)`

### 高级用户
1. 查看 [test_blind_aacnet.py](test_blind_aacnet.py) 源代码
2. 修改 `TestReport` 类添加自定义指标
3. 扩展模型定义以支持其他网络架构

---

## 💾 文件大小和性能

| 文件 | 大小 | 性能 |
|------|------|------|
| test_blind_aacnet.py | ~17KB | 高效（单张图像级处理） |
| aacnet_model.py | ~2KB | 轻量级包装 |
| launch_blind_test.py | ~10KB | 快速启动 |
| 文档 | ~50KB | 参考用 |

---

## ✅ 功能清单

### 支持的功能

- ✅ RGB图像处理（自动BGR↔RGB转换）
- ✅ 灰度转换（用于指标计算）
- ✅ 多种盲元坐标格式
  - CSV坐标文件（静态）
  - Mask图像文件
  - Frame级CSV（动态）
- ✅ 自动尺寸对齐
- ✅ 按组织结构组织结果
- ✅ CSV报告生成
- ✅ 多GPU支持
- ✅ 交互式和自动化两种运行方式
- ✅ 详细的参数验证和提示

### 评估指标

| 指标 | 适用范围 | 说明 |
|------|---------|------|
| PSNR | 全图 | 峰值信噪比 |
| SSIM | 全图 | 结构相似性 |
| blind_mae | 盲元区域 | 平均绝对误差 |
| blind_rmse | 盲元区域 | 均方根误差 |
| blind_psnr | 盲元区域 | 盲元PSNR |
| blind_mae_gain | 盲元区域 | 改进百分比 |

---

## 🔧 命令速查

### 最简单的使用
```bash
python launch_blind_test.py
```

### 一行命令测试
```bash
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint model.pth
```

### 完整配置
```bash
python test_blind_aacnet.py \
    --data_root /home/student_server/Qtt/NAFNet/data \
    --checkpoint model.pth \
    --save_dir ./results/v1 \
    --device cuda \
    --gpu_ids 0,1 \
    --image_border 0
```

---

## 📞 问题排查

### 找不到CSV文件
→ 检查 `test_mask/<group_name>/blind_coords.csv` 是否存在
→ 确保CSV有 x,y 列头

### 盲元指标为空
→ 检查盲元坐标是否正确加载
→ 查看命令行输出中的 "Loaded blind coords" 提示

### 显存不足
→ 使用 `--device cpu` 在CPU上运行
→ 当前脚本批大小固定为1，无法调整

### 模型加载失败
→ 检查checkpoint格式是否正确
→ 尝试用 `torch.load()` 直接检查

---

## 📝 文件清单速查

| 文件名 | 类型 | 行数 | 用途 |
|--------|------|------|------|
| test_blind_aacnet.py | Python | ~700 | 主测试脚本 |
| model/aacnet_model.py | Python | ~70 | 模型定义 |
| launch_blind_test.py | Python | ~300 | 启动工具 |
| run_blind_test.sh | Shell | ~100 | Shell脚本 |
| TEST_BLIND_GUIDE.md | 文档 | ~400行 | 详细指南 |
| BLIND_TEST_README.md | 文档 | ~350行 | 快速参考 |
| config_blind_test.template | 配置 | ~150行 | 参数模板 |
| INDEX.md | 文档 | 本文件 | 文件导航 |

---

**最后更新**: 2026-05-26

**下一步**: 选择一种启动方式开始测试！🚀
