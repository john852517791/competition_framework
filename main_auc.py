# pl_project/main.py

import os
import argparse
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping,LearningRateMonitor
import yaml # 导入 yaml 库
from lightning.pytorch.loggers import TensorBoardLogger
import importlib
from utils.tools.utils import convert_str_2_path
from config.config_class import GlobalConfig


def train(args:GlobalConfig,config_path):
    """主训练函数"""
    # 设置随机种子以保证可复现性
    pl.seed_everything(args.train.seed) # 从配置对象中访问参数
    data_util = importlib.import_module(convert_str_2_path(args.module.dataModule))
    tl_md = importlib.import_module(convert_str_2_path(args.module.tl_model_module))
    prj_model = importlib.import_module(convert_str_2_path(args.module.modelModule))
    
    # 1. 加载模型和处理器
    base_model = prj_model.Model(args)
    # 2. 设置 DataModule
    datamodule = data_util.MyDataModule(
        args = args.data,
        aug_args = args.augment
    )
    
    # 3. 设置 Lightning Model
    model = tl_md.tl_Model(
        model=base_model,
        args=args,
        config_path = config_path
    )

    # 4. 设置 Trainer 和 Callbacks
    checkpoint_callback = ModelCheckpoint( # 从配置对象中访问参数
        # dirpath=args.output_dir, 
        filename='best-checkpoint-{epoch:02d}-{val_auc:.4f}',
        save_top_k=args.train.save_top_k,
        verbose=True,
        monitor='val_auc',
        mode='max'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_auc',
        patience=args.train.patience,
        mode='max'
        ) # 从配置对象中访问参数

    logger = TensorBoardLogger(
        save_dir=args.logging.log_dir,
        name=args.logging.single_logname
        )
    lr_monitor = LearningRateMonitor(logging_interval='step') # 或 'epoch'

    trainer = pl.Trainer(
        max_epochs=args.train.epochs, # 从配置对象中访问参数
        accelerator=args.train.accelerator, # 从配置对象中访问参数
        devices=args.train.deviceid, # 从配置对象中访问参数
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=logger, # 从配置对象中访问参数
        log_every_n_steps=1,
    )

    # 5. 开始训练
    trainer.fit(model, datamodule=datamodule)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Lightning 图像分类训练脚本。")
    parser.add_argument('--config', type=str, default='config/config.yaml', help="配置文件路径。")
    cli_args = parser.parse_args()

    config = GlobalConfig.from_yaml(cli_args.config)
    train(config,cli_args.config)

if __name__ == '__main__':
    main()