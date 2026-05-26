import torch
import torch.nn as nn
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

        # 创建生成器
        self.net_G = aacnet.define_g(gpu_ids=opt.gpu_ids, image_size=(opt.image_height, opt.image_width))

        if self.isTrain:
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.net_G.parameters()),
                lr=opt.lr,
                betas=(getattr(opt, 'beta1', 0.5), getattr(opt, 'beta2', 0.9))
            )
            self.optimizers.append(self.optimizer_G)

        if not self.isTrain:
            self.setup(opt)
        else:
            # 如果是训练模式，也加载预训练模型
            self.setup(opt)

    def set_input(self, input_data, epoch=0):
        """Unpack input data from the data loader and perform necessary pre-process steps"""
        self.image_paths = input_data['img_path']
        
        # 获取输入图像和mask
        img = input_data['img']  # [B, 3, H, W], range [-1, 1]
        mask = input_data['mask']  # [B, 3, H, W], range [0, 1]

        if len(self.gpu_ids) > 0:
            img = img.cuda(self.gpu_ids[0])
            mask = mask.cuda(self.gpu_ids[0])

        # 保存原始值
        self.img_truth = img  # 原始图像 [-1, 1]
        self.mask = mask  # mask [0, 1]，1 表示有效区域，0 表示盲元区域
        
        # 应用mask：只保留有效区域的内容
        # mask 为 1 的位置保留原内容，为 0 的位置设为 0
        self.img_m = self.img_truth * self.mask  # 遮挡后的图像

    def test(self):
        """Forward function used in test time"""
        self.net_G.eval()
        
        # Generator 需要 mask 作为额外输入
        # 在 forward 时，我们需要适配 mask 的形状（从 [B, 3, H, W] 转换为 [B, 1, H, W]）
        mask_single = self.mask[:, 0:1, :, :]  # [B, 1, H, W]
        
        # 模型推理
        self.img_g, self.x_outs = self.net_G(self.img_m, mask_single)
        
        # 融合结果：在遮挡区域使用生成结果，在有效区域使用原始内容
        # mask 为 1 保留原内容，为 0 使用生成内容
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask

    def forward(self):
        """Run forward processing to get the inputs"""
        mask_single = self.mask[:, 0:1, :, :]
        self.img_g, self.x_outs = self.net_G(self.img_m, mask_single)
        self.img_out = self.img_g * (1 - self.mask) + self.img_truth * self.mask

    def optimize_parameters(self):
        """Optimize generator parameters for blind completion training"""
        self.forward()
        self.optimizer_G.zero_grad()
        self.loss_l1 = self.criterionL1(self.img_out, self.img_truth)
        self.loss_l1.backward()
        self.optimizer_G.step()

    def get_current_visuals(self):
        """Return visualization images"""
        visual_ret = {}
        visual_ret['img_m'] = util.tensor2im(self.img_m.data)
        visual_ret['img_truth'] = util.tensor2im(self.img_truth.data)
        visual_ret['img_out'] = util.tensor2im(self.img_out.data)
        return visual_ret
