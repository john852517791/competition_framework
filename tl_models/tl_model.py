# pl_project/tl_model.py

import torch
import torch.nn as nn
import lightning as pl
from utils.tools.loss_wrapper import get_loss_function
from utils.tools.optim_wrapper import get_optimizer
from utils.tools.scheduler_wrapper import get_scheduler
from config.config_class import GlobalConfig
from torchmetrics.classification import BinaryAUROC # For AUC
import shutil,os
import torch.nn.functional as F

class tl_Model(pl.LightningModule):
    """
    封装了模型、损失函数和优化逻辑的 LightningModule。
    """
    def __init__(self, model, args:GlobalConfig, config_path = "config/config.yaml"):
        super().__init__()
        self.args = args
        self.config_path = config_path
        self.model = model

        self.val_auc = BinaryAUROC()
        self.loss_fn = get_loss_function(args.train.loss_fn)
        # self.save_hyperparameters(args)

    def forward(self, x,train):
        feature,logits = self.model(x,train)
        return feature, logits


    def training_step(self, batch, batch_idx):
        x,labels = batch
        # if labels.dim() < 2:
        #     labels = labels.unsqueeze(1)
        _, logits = self(x,True)
        loss = self.loss_fn(logits, labels)
        # preds = torch.argmax(logits, dim=1)
        # acc = (preds == labels).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     x, labels = batch
    #     if labels.dim() < 2:
    #         labels = labels.unsqueeze(1)
    #     _, logits = self(x, True) # Assuming your forward returns hidden_state, logits
    #     loss = self.loss_fn(logits, labels)
    #     probs = F.softmax(logits, dim=-1)
    #     self.log('val_loss', loss, on_epoch=True, prog_bar=True)
    #     self.val_auc(probs, labels) # Accumulate probabilities and labels for AUC

    #     return {"loss": loss, "probs": probs, "labels": labels} # Return needed data for epoch_end

    # def on_validation_epoch_end(self):
    #     # Compute and log overall metrics at the end of the validation epoch
    #     epoch_auc = self.val_auc.compute()
    #     self.log('val_auc', epoch_auc, on_epoch=True, prog_bar=True)
    #     # Reset metrics for the next epoch
    #     self.val_auc.reset()

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
            pred_config_path = os.path.join(destination_config_dir, "pred.yaml")

            try:
                # 拷贝文件
                shutil.copy(source_config_path, destination_config_path)
                shutil.copy(source_config_path, pred_config_path)
                print(f"成功拷贝配置文件 '{source_config_path}' 到 '{destination_config_path}'")
            except Exception as e:
                print(f"拷贝配置文件失败：{e}")
        else:
            print("警告：Trainer 或其日志目录未初始化，无法拷贝配置文件。")

    def predict_step(self, batch, batch_idx):
        x, image_names = batch 
        _,logits = self(x,False) 
        if logits.shape[1]==2:
            probabilities = torch.softmax(logits, dim=1)
            scores = probabilities[:, 1] # Probability of the positive class
            # predicted_classes = torch.argmax(logits, dim=1)
        else:
            scores = logits.squeeze(1)
        # return {"image_name": image_names, "score": scores.tolist(), "predicted_class": predicted_classes.tolist()}
        return {"image_name": image_names, "score": scores.tolist()}

