import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from .base_model import BaseModel
from . import aacnet


class AACNetModel(BaseModel):
    """AACNet 盲元修复模型 — 使用原始 deformable attention 架构"""

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.loss_names = ['l1']
        self.visual_names = ['img_m', 'img_truth', 'img_out']
        self.model_names = ['G']
        self.isTrain = opt.isTrain
        self.use_amp = bool(getattr(opt, 'mixed_precision', False) and torch.cuda.is_available())
        self.scaler = GradScaler(enabled=self.use_amp)

        image_size = (getattr(opt, 'image_height', 512), getattr(opt, 'image_width', 640))
        self.net_G = aacnet.define_g(
            init_type=getattr(opt, 'init_type', 'normal'),
            gpu_ids=opt.gpu_ids,
            image_size=image_size
        )

        if self.isTrain:
            self.criterionL1 = nn.L1Loss()
            lr_val = getattr(opt, 'lr', None) or getattr(opt, 'learning_rate', 0.0001)
            self.optimizer_G = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.net_G.parameters()),
                lr=lr_val,
                betas=(getattr(opt, 'beta1', 0.5), getattr(opt, 'beta2', 0.9))
            )
            self.optimizers.append(self.optimizer_G)
            self.setup(opt)

    def set_input(self, input_data, epoch=0):
        if 'blur' not in input_data:
            raise KeyError("set_input requires 'blur' key with the corrupted input image")
        if 'sharp' not in input_data:
            raise KeyError("set_input requires 'sharp' key with the clean ground truth image")
        if 'mask' not in input_data:
            raise KeyError("set_input requires 'mask' key with the blind pixel mask")

        self.image_paths = input_data['img_path']

        device = f'cuda:{self.gpu_ids[0]}' if self.gpu_ids else 'cpu'
        self.img_m = input_data['blur'].to(device)
        self.img_truth = input_data['sharp'].to(device)
        self.mask = input_data['mask'].to(device)

    def forward(self):
        mask_single = self.mask[:, 0:1, :, :]  # [B, 1, H, W]
        self.img_g, self.x_outs = self.net_G(self.img_m, mask_single)
        # 盲元区域(0)用生成结果，有效区域(1)保留真值
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask

    def test(self):
        self.net_G.eval()
        with torch.no_grad():
            self.forward()

    def optimize_parameters(self):
        self.optimizer_G.zero_grad(set_to_none=True)
        with autocast(enabled=self.use_amp):
            self.forward()
            self.loss_l1 = self.criterionL1(self.img_out, self.img_truth)

        if self.use_amp:
            self.scaler.scale(self.loss_l1).backward()
            self.scaler.step(self.optimizer_G)
            self.scaler.update()
        else:
            self.loss_l1.backward()
            self.optimizer_G.step()
