import argparse
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
import importlib
from utils.tools.utils import convert_str_2_path
from config.config_class import GlobalConfig
import pandas as pd
from lightning import LightningModule


def predict(args: GlobalConfig, config_path, checkpoint_path: str, output_csv_path):
    """主预测函数"""
    # 1. 加载 DataModule
    data_util = importlib.import_module(convert_str_2_path(args.module.dataModule))
    datamodule = data_util.MyDataModule(
        args=args.data,
        aug_args=args.augment
    )
    prj_model = importlib.import_module(convert_str_2_path(args.module.modelModule))
    base_model = prj_model.Model(args)

    # 2. 加载 Lightning Model (从 checkpoint 加载)
    tl_md = importlib.import_module(convert_str_2_path(args.module.tl_model_module))
    # LightningModule.load_from_checkpoint()
    model = tl_md.tl_Model.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model = base_model,
        args=args,
        config_path=config_path,
        map_location = "cuda"
        # strict=False # 如果模型结构有细微变化，可以设置为 False
    )
    model.eval() # 设置模型为评估模式
    
    # 3. 设置 Trainer
    trainer = pl.Trainer(
        accelerator=args.train.accelerator,
        devices=args.train.deviceid,
        logger=False # 预测时通常不需要 logger
    )

    # 4. 开始预测
    print(f"Starting prediction using checkpoint: {checkpoint_path}")
    predictions_output = trainer.predict(model, datamodule=datamodule)
    if trainer.is_global_zero:
        all_image_names = []
        all_scores = []
        # all_predicted_classes = []

        for batch_result in predictions_output:
            all_image_names.extend(batch_result["image_name"])
            all_scores.extend(batch_result["score"])
            # all_predicted_classes.extend(batch_result["predicted_class"])

        # Create a DataFrame
        results_df = pd.DataFrame({
            "image_name": all_image_names,
            "score": all_scores,
            # "predicted_class": all_predicted_classes
        })
        # 去重
        results_df_deduplicated = results_df.drop_duplicates(subset=['image_name'], keep='first')

        # Save to CSV
        results_df_deduplicated.to_csv(output_csv_path, index=False, header=True)
        print(f"Prediction results saved to {output_csv_path}")



def main():
    ckpname = "a_logs/vit_base_patch16_pretrained_aasist/version_2/checkpoints/best-checkpoint-epoch=61-train_loss=0.0533.ckpt"
    # "a_logs/mae_vit_base_patch14_pretrained_attenf_2/version_1/checkpoints/best-checkpoint-epoch=59-train_loss=0.1969.ckpt"
    path = ckpname.split("checkpoints")[0]
    parser = argparse.ArgumentParser(description="PyTorch Lightning 图像分类训练和预测脚本。")
    parser.add_argument('--config', type=str, default=f'{path}/pred.yaml', help="配置文件路径。")
    parser.add_argument('--output_csv_path', type=str, default=f'{path}/submit.csv')
    parser.add_argument('--checkpoint_path', type=str, default=f"{ckpname}",
                        help="预测模式下加载的模型 checkpoint 路径。")
    cli_args = parser.parse_args()

    config:GlobalConfig = GlobalConfig.from_yaml(cli_args.config)
    print(config.train.deviceid)
    predict(config, cli_args.config, cli_args.checkpoint_path,cli_args.output_csv_path)

if __name__ == '__main__':
    main()