# pl_project/tl_model.py

import torch
import torch.nn as nn
import lightning as pl
from utils.tools.loss_wrapper import get_loss_function
from utils.tools.optim_wrapper import get_optimizer
from utils.tools.scheduler_wrapper import get_scheduler
from config.config_class import GlobalConfig
import shutil,os

class tl_Model(pl.LightningModule):
    """
    封装了模型、损失函数和优化逻辑的 LightningModule。
    """
    def __init__(self, model, args:GlobalConfig, config_path = "config/config.yaml"):
        super().__init__()
        self.args = args
        self.config_path = config_path
        self.model = model

        self.loss_fn = get_loss_function(args.train.loss_fn)
        # self.save_hyperparameters(args)

    def forward(self, x,train):
        feature,logits = self.model(x,train)
        return feature, logits


    def training_step(self, batch, batch_idx):
        x,labels = batch
        hidden_state, logits = self(x,True)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,labels = batch
        hidden_state, logits = self(x,True)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        
        self.optimizer = get_optimizer(self.model,self.args.train)
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            config_train=self.args.train,
        )
        return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": self.scheduler["scheduler"],
                    "monitor": "train_loss",
                }
            }
        
    def on_train_start(self):
        """
        在训练开始时（Trainer 设置好日志目录后）拷贝配置文件。
        """
        # 确保 trainer 和其 log_dir 已经初始化
        if self.trainer and self.trainer.log_dir:
            source_config_path = self.config_path # 配置文件在项目中的路径
            
            # 检查源配置文件是否存在
            if not os.path.exists(source_config_path):
                print(f"警告：找不到源配置文件 '{source_config_path}'，跳过拷贝。")
                return

            destination_config_dir = self.trainer.log_dir
            destination_config_path = os.path.join(destination_config_dir, os.path.basename(source_config_path))

            try:
                # 拷贝文件
                shutil.copy(source_config_path, destination_config_path)
                print(f"成功拷贝配置文件 '{source_config_path}' 到 '{destination_config_path}'")
            except Exception as e:
                print(f"拷贝配置文件失败：{e}")
        else:
            print("警告：Trainer 或其日志目录未初始化，无法拷贝配置文件。")

