# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AACNet is a PyTorch-based deep learning model for **infrared image blind pixel restoration**. It takes degraded grayscale IR images with blind/flash pixel defects and outputs restored results, guided by a per-pixel mask indicating defective regions.

## Commands

```bash
# Training
python main.py --train --config_path ./experiment.cfg

# Testing (loads best_model.pt)
python main.py --test --config_path ./experiment.cfg

# Standalone test
python ./test.py --config_path ./experiment.cfg

# Install dependencies (PyTorch 1.8.1 + CUDA 11.1)
pip install -r requirements.txt
```

No linting or test suite exists in this repository.

## Configuration

All behavior is driven by `experiment.cfg` (ConfigParser INI format). The `ConfigLoader` in `util/config_loader.py` reads it into an `argparse.Namespace`. Key sections:

- `[dataset]` — paths to `train_blur/sharp/mask`, `val_*`, `test_*` under `data_root`. Image size 640×512.
- `[training]` — `batch_size=4`, `learning_rate=0.0001`, `num_epochs=200`, `val_interval=5`, cosine LR schedule.
- `[checkpoint]` — `checkpoints_dir`, `model_prefix=best_model`, `best_metric=psnr`.
- `[device]` — `gpu_ids=0`, `mixed_precision=True`, single-GPU only (`use_dataparallel=False`).
- `[blind_pixel]` — `mask_type=both`, `static_coords_file=blind_pixel_coords.csv`, `dynamic_coords_file=flash_pixel_coords.csv`.

## Architecture

### Data pipeline (`dataloader/blind_pixel_loader.py`)

`BlindPixelDataset` loads images organized in **group subdirectories** (`001/`, `002/`, etc.) under `{phase}_blur/`, `{phase}_sharp/`, `{phase}_mask/`. Each group dir contains:

- Images (`.png`) in blur and sharp dirs, paired by filename.
- Mask dir contains `blind_pixel_coords.csv` (static defective coordinates) and optionally `blind_pixel_mask.png` (fallback mask image) and `flash_pixel_coords.csv` (per-frame dynamic flash coordinates).

The dataset constructs per-image masks by combining static blind pixel coordinates with that frame's dynamic flash pixels. Mask semantics: **1 = valid pixel, 0 = defective pixel to restore**. Images are normalized to `[-1, 1]`, masks stay in `[0, 1]`.

Training applies synchronized data augmentation (horizontal flip, small rotation) to image+mask together.

`create_dataloader()` is the factory function; always use this rather than constructing the dataset/dataloader directly for a given phase.

### Model (`model/`)

**Model factory** (`model/__init__.py`): `create_model(opt)` auto-imports `model.{model_name}_model` and finds the subclass of `BaseModel` whose name starts with `model_name`. Currently loads `model.aacnet_model.AACNetBlind`.

**`AACNetBlind`** (`model/aacnet_model.py`): Extends `BaseModel`. Key behaviors:
- `set_input()` unpacks `blur`, `sharp`, `mask` tensors from the dataloader, stores them as `self.img_m` (input), `self.img_truth` (ground truth), `self.mask`.
- `forward()` runs the generator then blends output with ground truth: `output = gen_output * (1 - mask) + truth * mask`. This ensures only defective regions are generated; valid regions are passed through.
- `optimize_parameters()` uses L1 loss with AMP mixed precision (`GradScaler`).
- `test()` sets model to eval mode and runs the same forward path.

**Generator** (`model/aacnet.py`): U-Net-like architecture with 3 downsampling stages:
1. `ResBlock0_v2` (4→48 channels, 5×5 conv) + `ResBlock_v2` at full resolution
2. Downsample to 1/2 (96ch), 3× ResBlock; Downsample to 1/4 (192ch), 4× ResBlock; Downsample to 1/8 (384ch)
3. Bottleneck: ResBlock → 4× `DAttentionBaseline_gate_factor` (deformable attention) interleaved with ResBlocks. Feature map at this stage is `(H/8, W/8)`, so image_size must be divisible by 8.
4. Decoder: upsample + skip connection fusion + ResBlocks, with intermediate outputs at 1/4, 1/2, 1/1 scales (L32, L64, L128, L256). Final output: `ReflectionPad2d(3)` + 7×7 conv → tanh.

Input is 4-channel: `[blur_image (3ch), mask (1ch)]`. Gated convolutions (`GatedConv`) are used in all ResBlocks. InstanceNorm throughout, LeakyReLU(0.2) activations.

**`DAttentionBaseline_gate_factor`** (`model/dat_blocks.py`): Deformable attention module. Uses group-wise learned offsets via gated convolution, bilinear sampling for key/value tokens, and optional relative position encoding. The offset range factor is a learnable parameter.

**`BaseModel`** (`model/base_model.py`): Provides `loss_names`, `visual_names`, `model_names` lists, `get_current_errors()`, `get_current_visuals()`, `save_networks()`/`load_networks()`.

### Training flow (`main.py`)

`train()` orchestrates: data loading → model creation → `CheckpointManager` initialization → epoch loop with `train_epoch()` and periodic `validate()`. Resumption from `last_state.pt` is supported via `[resume]` config.

`validate()`: runs `model.test()`, computes PSNR/SSIM via `MetricCalculator`, logs to `validation.txt`. Best models (by `best_metric`, default `psnr`) update `best_model.pt`.

### Testing flow (`test.py`)

`run_test()` loads `best_model.pt`, runs inference over the test set, computes both **global** metrics (PSNR, SSIM) and **blind-pixel-specific** metrics (Blind MAE, Blind RMSE, Blind PSNR) — the latter computed only over pixels marked in `blind_pixel_coords.csv` + `flash_pixel_coords.csv`. Results are saved per-group as CSVs under `results_dir`.

### Checkpoint management (`util/checkpoint_manager.py`)

Only two files are maintained: `best_model.pt` (best validation metric) and `last_state.pt` (latest training state for resume). Both contain model state, optimizer state, scheduler state, metrics, and epoch. Legacy `.pth` files are cleaned up.

### Key utilities

- `util/metrics.py` — `MetricCalculator` for PSNR, SSIM, MAE, RMSE (supports crop_border).
- `util/config_loader.py` — `ConfigLoader` reads `.cfg` INI files into `Namespace` objects with typed getters.
- `util/logger.py` — Writes training/validation logs to text files.
- `util/util.py` — `tensor2im()` (tensor → numpy uint8), image I/O helpers.

## Data directory convention

```
data_new/
├── train_blur/{001..NNN}/*.png
├── train_sharp/{001..NNN}/*.png
├── train_mask/{001..NNN}/blind_pixel_coords.csv, flash_pixel_coords.csv
├── val_blur/{001..NNN}/*.png
├── val_sharp/{001..NNN}/*.png
├── val_mask/{001..NNN}/blind_pixel_coords.csv, flash_pixel_coords.csv
├── test_blur/{001..NNN}/*.png
├── test_sharp/{001..NNN}/*.png
└── test_mask/{001..NNN}/blind_pixel_coords.csv, flash_pixel_coords.csv
```

Images are 640×512 grayscale PNGs. Mask CSV columns: `x`, `y` (pixel coordinates). Flash CSV also has `frame_name` column for per-image lookup.

## Important constraints

- Image dimensions must be divisible by 8 (3 downsampling stages).
- The model is designed for single-GPU (`gpu_ids=0`). DataParallel is explicitly disabled.
- Training mask only uses `blind_pixel_coords.csv` (static defects). Validation/testing blind metrics also include `flash_pixel_coords.csv` (dynamic defects) for evaluation only.
- Input images are grayscale but the pipeline treats them as 3-channel RGB throughout.
