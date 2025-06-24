# pl_project/tl_model.py

import torch
import torch.nn as nn
import lightning as pl

class MyLightningModel(pl.LightningModule):
    """
    封装了模型、损失函数和优化逻辑的 LightningModule。
    """
    def __init__(self, model, num_classes: int = 2, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # 为预训练模型添加一个分类头
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, self.num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        # 保存超参数，方便后续加载
        self.save_hyperparameters(ignore=['model'])

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        # 对于 Vision Transformer (ViT) 模型，我们使用 pooler_output
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output)
        return logits

    def _common_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

