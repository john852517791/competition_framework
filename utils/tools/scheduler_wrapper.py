from torch.optim import lr_scheduler
import torch
from config.config_class import TrainConfig
import transformers

def get_scheduler(optimizer: torch.optim.Optimizer, config_train:TrainConfig):
    """
    根據配置返回一個 PyTorch 學習率排程器實例。

    Args:
        optimizer (torch.optim.Optimizer): 要關聯的優化器實例。
        config_train: 包含排程器配置的對象 (例如 GlobalConfig.train)。
                      需要有 .lr_scheduler, .lr_scheduler_step_size, .lr_scheduler_gamma,
                      .lr_scheduler_mode, .lr_scheduler_patience, .lr_scheduler_factor,
                      .lr_scheduler_t_max 屬性。
        total_epochs (int, optional): 訓練的總 epochs 數，對於 CosineAnnealingLR 等可能需要。

    Returns:
        dict or None: 返回一個 PyTorch Lightning 兼容的排程器字典，或 None 如果不使用排程器。

    Raises:
        ValueError: 如果提供不支援的排程器名稱或缺少必要參數。
    """
    total_epochs = config_train.epochs
    scheduler_name = config_train.lr_scheduler.lower()
    
    scheduler_config = {
        'interval': 'step', # 預設每 epoch 更新
        'frequency': 1,
    }

    if scheduler_name == 'steplr':
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=config_train.lr_scheduler_step_size, 
            gamma=config_train.lr_scheduler_gamma
        )
        scheduler_config['scheduler'] = scheduler
    elif scheduler_name == 'reduceonplateau' or scheduler_name == 'reducelronplateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=config_train.lr_scheduler_mode, # 'min' or 'max'
            factor=config_train.lr_scheduler_factor,
            patience=config_train.lr_scheduler_patience
        )
        scheduler_config['scheduler'] = scheduler
        scheduler_config['monitor'] = 'val_loss' # 監控驗證損失，以便在 plateau 時調整學習率
        scheduler_config['interval'] = 'epoch'
        scheduler_config['frequency'] = 1
    elif scheduler_name == 'cosineannealinglr':
        if total_epochs is None:
            raise ValueError("使用 CosineAnnealingLR 時，必須提供 `total_epochs` 參數。")
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs
            # T_max=config_train.lr_scheduler_t_max if config_train.lr_scheduler_t_max else total_epochs
        )
        scheduler_config['scheduler'] = scheduler

    elif scheduler_name == 'linearwarmupcosineannealinglr':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer = optimizer, 
            num_warmup_steps=(662810/config_train.batch_size) * config_train.warmup_steps,          
            num_training_steps = (662810/config_train.batch_size) * config_train.epochs
        )
        scheduler_config['scheduler'] = scheduler

    elif scheduler_name == 'none':
        return None # 不使用排程器
    else:
        raise ValueError(f"不支援的學習率排程器: {scheduler_name}。請選擇 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'None'。")
    
    return scheduler_config

# 範例使用 (在你的 LightningModule 的 configure_optimizers 中)
# optimizer = get_optimizer(self.model, self.args.train)
# scheduler_dict = get_scheduler(optimizer, self.args.train, total_epochs=self.args.train.epochs)
# return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict} if scheduler_dict else optimizer