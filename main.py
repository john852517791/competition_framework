# pl_project/main.py

import os
import argparse
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping 
import yaml # 导入 yaml 库
from lightning.pytorch.loggers import WandbLogger
from data_loader.dataloader import load_model_and_processor
from data_loader.datamodule import ImageDataModule
from tl_models.tl_model import MyLightningModel

# 定义一个简单的类，用于将字典转换为可以通过属性访问的对象
class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def train(args):
    """主训练函数"""
    # 设置随机种子以保证可复现性
    pl.seed_everything(args.seed) # 从配置对象中访问参数

    # 1. 加载模型和处理器
    base_model, processor = load_model_and_processor(args.model_name_or_path) # 从配置对象中访问参数

    # 2. 设置 DataModule
    datamodule = ImageDataModule(
        data_dir=args.data_dir, # 从配置对象中访问参数
        processor=processor,
        batch_size=args.batch_size, # 从配置对象中访问参数
        num_workers=args.num_workers # 从配置对象中访问参数
    )
    # 重要: 调用 setup() 来初始化数据集和类别映射
    datamodule.setup()
    
    # 3. 设置 Lightning Model
    num_classes = len(datamodule.class_to_idx)
    model = MyLightningModel(
        model=base_model,
        num_classes=num_classes,
        learning_rate=args.learning_rate # 从配置对象中访问参数
    )

    # 4. 设置 Trainer 和 Callbacks
    checkpoint_callback = ModelCheckpoint( # 从配置对象中访问参数
        # dirpath=args.output_dir, 
        filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min') # 从配置对象中访问参数


    # 使用 WandbLogger 替换 TensorBoardLogger
    # 设置 log_model=False, 这样 logger 就不会上传模型权重，只监控指标。
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_run_name, # 如果为 None，wandb 会自动生成一个
        save_dir=args.output_dir,
        log_model=False)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs, # 从配置对象中访问参数
        accelerator=args.accelerator, # 从配置对象中访问参数
        devices=args.devices, # 从配置对象中访问参数
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger, # 从配置对象中访问参数
        log_every_n_steps=10,
    )

    # 5. 开始训练
    trainer.fit(model, datamodule)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Lightning 图像分类训练脚本。")
    parser.add_argument('--config', type=str, default='config/config.yaml', help="配置文件路径。")
    cli_args = parser.parse_args()

    # 从 YAML 文件加载配置
    with open(cli_args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 将配置字典转换为对象，以便通过属性访问
    config = Config(config_dict)

    os.makedirs(config.output_dir, exist_ok=True) # 从配置对象中访问参数
    train(config)

if __name__ == '__main__':
    main()