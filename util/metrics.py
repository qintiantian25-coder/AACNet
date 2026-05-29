"""
指标计算工具 - 计算PSNR、SSIM等评估指标
"""

import numpy as np
import cv2


class MetricCalculator:
    """指标计算类"""
    
    def __init__(self, crop_border=0):
        """
        初始化指标计算器
        
        Args:
            crop_border: 计算时的边界裁剪像素数
        """
        self.crop_border = crop_border
    
    def calculate_psnr(self, img1, img2):
        """
        计算PSNR
        
        Args:
            img1: 预测图像 (uint8 或 float32 [0,255])
            img2: 目标图像 (uint8 或 float32 [0,255])
        
        Returns:
            psnr: PSNR值 (dB)
        """
        # 裁剪边界
        if self.crop_border > 0:
            img1 = img1[self.crop_border:-self.crop_border, 
                        self.crop_border:-self.crop_border, :]
            img2 = img2[self.crop_border:-self.crop_border,
                        self.crop_border:-self.crop_border, :]
        
        # 转换为float32
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # 计算MSE
        mse = np.mean((img1 - img2) ** 2)
        
        if mse < 1e-10:
            return 100.0  # 图像完全相同
        
        # 计算PSNR
        psnr = 10 * np.log10(255.0 * 255.0 / mse)
        
        return psnr
    
    def calculate_ssim(self, img1, img2):
        """
        计算SSIM (Structural Similarity Index)
        
        Args:
            img1: 预测图像 (uint8 或 float32 [0,255])
            img2: 目标图像 (uint8 或 float32 [0,255])
        
        Returns:
            ssim: SSIM值 (0-1)
        """
        # 裁剪边界
        if self.crop_border > 0:
            img1 = img1[self.crop_border:-self.crop_border,
                        self.crop_border:-self.crop_border, :]
            img2 = img2[self.crop_border:-self.crop_border,
                        self.crop_border:-self.crop_border, :]
        
        # 转换为float32
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        # SSIM计算常数
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        # 计算均值
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        # 计算方差和协方差
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return np.mean(ssim_map)

    def calculate_psnr_ssim(self, img1, img2):
        """一次性计算 PSNR 和 SSIM，保证训练验证与测试走同一套实现。"""
        return self.calculate_psnr(img1, img2), self.calculate_ssim(img1, img2)
    
    def calculate_mae(self, img1, img2):
        """
        计算MAE (Mean Absolute Error)
        
        Args:
            img1: 预测图像
            img2: 目标图像
        
        Returns:
            mae: MAE值
        """
        if self.crop_border > 0:
            img1 = img1[self.crop_border:-self.crop_border,
                        self.crop_border:-self.crop_border, :]
            img2 = img2[self.crop_border:-self.crop_border,
                        self.crop_border:-self.crop_border, :]
        
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        mae = np.mean(np.abs(img1 - img2))
        
        return mae
    
    def calculate_rmse(self, img1, img2):
        """
        计算RMSE (Root Mean Square Error)
        
        Args:
            img1: 预测图像
            img2: 目标图像
        
        Returns:
            rmse: RMSE值
        """
        if self.crop_border > 0:
            img1 = img1[self.crop_border:-self.crop_border,
                        self.crop_border:-self.crop_border, :]
            img2 = img2[self.crop_border:-self.crop_border,
                        self.crop_border:-self.crop_border, :]
        
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)
        
        mse = np.mean((img1 - img2) ** 2)
        rmse = np.sqrt(mse)
        
        return rmse
