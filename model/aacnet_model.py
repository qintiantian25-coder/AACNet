import torch
import torch.nn as nn
from .base_model import BaseModel
from . import network 

class AACNetModel(BaseModel):
    """
    AACNet 接口桥接版 - 专用于对比实验
    修复了验证阶段属性读取报错，保留了梯度裁剪与标准 L1 监督。
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='blind_pixel')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_L1', 'G_Mask', 'G_Total']
        self.visual_names = ['img_m', 'img_recon', 'img_truth', 'mask']
        self.model_names = ['G']

        # 初始化生成器
        init_type = getattr(opt, 'init_type', 'normal')
        self.netG = network.define_g(init_type, self.gpu_ids)
        
        # 兼容父类命名规范
        self.net_G = self.netG 
        
        # 挂载组件
        self.netG.encoder = network.Encoder()
        self.netG.decoder = network.Decoder()
        
        if len(self.gpu_ids) > 0:
            self.netG.to(f'cuda:{self.gpu_ids[0]}')

        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()
            lr = getattr(opt, 'lr', 0.0001)
            beta1 = getattr(opt, 'beta1', 0.5)
            beta2 = getattr(opt, 'beta2', 0.999)
            weight_decay = getattr(opt, 'weight_decay', 0)
            
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
            )
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input_data, epoch=0):
        if 'blur' not in input_data and 'img' in input_data:
            input_data['blur'] = input_data['img']
        if 'sharp' not in input_data:
            input_data['sharp'] = input_data.get('blur', torch.zeros(1)).clone()
        if 'mask' not in input_data:
            input_data['mask'] = torch.ones_like(input_data['blur'][:, :1, :, :])

        device = f'cuda:{self.gpu_ids[0]}' if self.gpu_ids else 'cpu'
        self.img_m = input_data['blur'].to(device)
        self.img_truth = input_data['sharp'].to(device)
        self.mask = input_data['mask'].to(device)

    def forward(self):
        features, masks = self.netG.encoder(self.img_m, self.mask)
        self.img_recon = self.netG.decoder(features, masks)
        # 💡 修复验证阶段属性读取错误
        self.img_out = self.img_recon

    def test(self):
        """
        处理验证/测试阶段的推理逻辑
        """
        with torch.no_grad():
            self.forward()

    def backward_G(self):
        # 全局 L1 损失
        self.loss_G_L1 = self.criterionL1(self.img_recon, self.img_truth)
        
        # 盲元区域 L1 损失 (系数 1.0)
        blind_zone_recon = self.img_recon * (1.0 - self.mask)
        blind_zone_truth = self.img_truth * (1.0 - self.mask)
        self.loss_G_Mask = self.criterionL1(blind_zone_recon, blind_zone_truth) * 1.0 
        
        self.loss_G_Total = self.loss_G_L1 + self.loss_G_Mask
        self.loss_G_Total.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        
        # 梯度裁剪：防止训练不稳定性
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
        
        self.optimizer_G.step()

    # 兼容 checkpoint 机制
    def state_dict(self):
        return self.netG.state_dict()

    def load_state_dict(self, state_dict):
        self.netG.load_state_dict(state_dict)