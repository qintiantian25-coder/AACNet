"""
日志记录器 - 负责训练日志的记录
"""

import os
from datetime import datetime


class Logger:
    """日志记录类"""
    
    def __init__(self, config, log_name='training'):
        """
        初始化日志记录器
        
        Args:
            config: 配置对象
            log_name: 日志文件前缀，例如 training / validation
        """
        self.config = config
        self.log_dir = config.log_dir
        self.log_file = None
        self.log_name = log_name
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建固定名称的日志文件（训练与验证各一个），覆盖旧文件
        log_filename = f'{self.log_name}.txt'
        self.log_path = os.path.join(self.log_dir, log_filename)

        # 以写模式打开，确保每次运行生成的仅有最新的两个日志文件
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # 记录初始信息
        self.log("="*60)
        self.log(f"AACNet 盲元补完网络 - {self.log_name}日志")
        self.log("="*60)
        self.log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"日志文件: {self.log_path}")
        self.log("="*60)
        self.log("")
    
    def log(self, message):
        """
        记录日志信息
        
        Args:
            message: 要记录的消息
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # 同时输出到控制台和文件
        print(message)
        
        if self.log_file:
            self.log_file.write(log_message + '\n')
            self.log_file.flush()
    
    def log_config(self, config):
        """记录配置信息"""
        self.log("\n配置信息:")
        self.log("-"*60)
        
        # 记录关键配置
        self.log(f"  数据集根目录: {config.data_root}")
        self.log(f"  模型: {config.model_name}")
        self.log(f"  训练轮数: {config.num_epochs}")
        self.log(f"  批次大小: {config.batch_size}")
        self.log(f"  学习率: {config.learning_rate}")
        self.log(f"  优化器: {config.optimizer_type}")
        self.log(f"  验证间隔: {config.val_interval}")
        self.log(f"  检查点目录: {config.checkpoint_dir}")
        self.log(f"  设备: GPU {config.gpu_ids}")
        self.log("-"*60 + "\n")
    
    def log_epoch_start(self, epoch, num_epochs):
        """记录epoch开始"""
        self.log(f"\n[Epoch {epoch+1}/{num_epochs}] 开始训练...")
    
    def log_epoch_end(self, epoch, train_loss, lr):
        """记录epoch结束"""
        self.log(f"[Epoch {epoch+1}] 完成 - 训练损失: {train_loss:.4f}, 学习率: {lr:.6f}")
    
    def log_validation(self, epoch, metrics):
        """记录验证结果"""
        self.log(f"[Epoch {epoch+1}] 验证结果:")
        for key, value in metrics.items():
            self.log(f"  {key}: {value:.4f}")
    
    def log_best_model(self, epoch, metric_name, metric_value):
        """记录最佳模型信息"""
        self.log(f"✓ 保存最佳模型 (Epoch {epoch+1})")
        self.log(f"  {metric_name}: {metric_value:.4f}")
    
    def log_test_start(self):
        """记录测试开始"""
        self.log("\n" + "="*60)
        self.log("开始测试")
        self.log("="*60)
    
    def log_test_result(self, avg_psnr, avg_ssim):
        """记录测试结果"""
        self.log(f"\n测试结果:")
        self.log(f"  平均 PSNR: {avg_psnr:.4f}")
        self.log(f"  平均 SSIM: {avg_ssim:.4f}")
    
    def save_and_close(self):
        """保存并关闭日志文件"""
        if self.log_file:
            self.log("\n" + "="*60)
            self.log(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.log("="*60)
            
            self.log_file.close()
            self.log_file = None
            
            print(f"\n日志已保存到: {self.log_path}")
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        if self.log_file:
            self.log_file.close()
