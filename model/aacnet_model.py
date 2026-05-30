import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from model.base_model import BaseModel
from model import aacnet
from util import util
import os


class AACNetBlind(BaseModel):
    """AACNet 模型用于盲元补完测试"""

    def name(self):
        return "AACNet Blind Completion"

    @staticmethod
    def modify_options(parser, is_train=True):
        """Add new options and rewrite default values for existing options"""
        parser.add_argument('--ngf', type=int, default=48, help='# of gen filters in the first conv layer')
        return parser

    def __init__(self, opt):
        """Initial the AACNet blind model"""
        BaseModel.__init__(self, opt)

        self.loss_names = ['l1']
        self.visual_names = ['img_m', 'img_truth', 'img_out']
        self.model_names = ['G']
        self.isTrain = opt.isTrain
        self.use_amp = bool(getattr(opt, 'mixed_precision', False) and torch.cuda.is_available())
        self.scaler = GradScaler(enabled=self.use_amp)

        # 创建生成器
        self.net_G = aacnet.define_g(gpu_ids=opt.gpu_ids, image_size=(opt.image_height, opt.image_width))

        if self.isTrain:
            self.criterionL1 = nn.L1Loss()
            # 防御性读取：若 opt 里是 learning_rate 则兼容
            lr_val = opt.lr if hasattr(opt, 'lr') else getattr(opt, 'learning_rate', 0.0001)
            self.optimizer_G = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.net_G.parameters()),
                lr=lr_val,
                betas=(getattr(opt, 'beta1', 0.5), getattr(opt, 'beta2', 0.9))
            )
            self.optimizers.append(self.optimizer_G)

        if self.isTrain:
            # 训练模式只初始化优化器与调度器，不自动尝试加载不存在的测试权重
            self.setup(opt)

    def set_input(self, input_data, epoch=0):
        """
        从数据加载器中解包输入数据
        已完美适配重构后的 blind_pixel_loader 键名
        """
        self.image_paths = input_data['img_path']
        
        # 完美对接 Dataloader 返回的键名
        img_blur = input_data['blur']    # [B, 3, H, W], range [-1, 1] (含盲元/闪元图)
        img_sharp = input_data['sharp']  # [B, 3, H, W], range [-1, 1] (干净真值图)
        mask = input_data['mask']        # [B, 1, H, W] 或 [B, 3, H, W], range [0, 1]

        if len(self.gpu_ids) > 0:
            img_blur = img_blur.cuda(self.gpu_ids[0])
            img_sharp = img_sharp.cuda(self.gpu_ids[0])
            mask = mask.cuda(self.gpu_ids[0])

        self.img_truth = img_sharp  # 干净的真值目标 [-1, 1]
        self.mask = mask            # mask [0, 1]，1 表示有效区域，0 表示盲元区域
        
        # 【关键修正】直接使用 Dataloader 已经处理好盲元/闪元的真实图像输入
        # 不再在 [-1, 1] 的特征上直接乘以 mask（乘 0 会把像素变成中间色灰色，导致特征断层）
        self.img_m = img_blur

    def test(self):
        """Forward function used in test time"""
        self.net_G.eval()
        
        # Generator 需要单通道 mask 作为额外输入
        mask_single = self.mask[:, 0:1, :, :]  # 确保是 [B, 1, H, W]
        
        # 模型推理
        self.img_g, self.x_outs = self.net_G(self.img_m, mask_single)
        
        # 融合结果：在遮挡区域(0)使用生成结果，在有效区域(1)使用原始内容
        # mask 为 1 保留原内容，为 0 使用生成内容
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask

    def forward(self):
        """Run forward processing to get the inputs"""
        mask_single = self.mask[:, 0:1, :, :]
        self.img_g, self.x_outs = self.net_G(self.img_m, mask_single)
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask

    def optimize_parameters(self):
        """Optimize generator parameters for blind completion training"""
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

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = {}
        visual_ret['img_m'] = util.tensor2im(self.img_m.data)
        visual_ret['img_truth'] = util.tensor2im(self.img_truth.data)
        visual_ret['img_out'] = util.tensor2im(self.img_out.data)
        return visual_ret