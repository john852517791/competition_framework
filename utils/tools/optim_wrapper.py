import torch.optim as optim
import torch.nn as nn # 為了模型參數
from config.config_class import TrainConfig

def get_optimizer(model: nn.Module, config_train:TrainConfig):
    """
    根據配置返回一個 PyTorch 優化器實例。

    Args:
        model (nn.Module): 要優化的模型。
        config_train: 包含優化器配置的對象 (例如 GlobalConfig.train)。
                      需要有 .optimizer, .learning_rate, .weight_decay, .momentum 屬性。

    Returns:
        torch.optim.Optimizer: 優化器實例。

    Raises:
        ValueError: 如果提供不支援的優化器名稱。
    """
    optimizer_name = config_train.optimizer.lower()
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config_train.learning_rate, 
            weight_decay=config_train.weight_decay
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config_train.learning_rate, 
            weight_decay=config_train.weight_decay
        )
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config_train.learning_rate, 
            weight_decay=config_train.weight_decay, 
            momentum=config_train.momentum
        )
    else:
        raise ValueError(f"不支援的優化器: {optimizer_name}。請選擇 'Adam', 'AdamW', 'SGD'。")
    
    return optimizer

# 範例使用 (在你的 LightningModule 的 configure_optimizers 中)
# optimizer = get_optimizer(self.model, self.args.train)
