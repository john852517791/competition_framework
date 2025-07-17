import torch.nn as nn
import torch
def get_loss_function(loss_name: str, **kwargs):
    """
    根據名稱返回一個 PyTorch 損失函數實例。

    Args:
        loss_name (str): 損失函數的名稱 (不區分大小寫)。
                         支援: 'CrossEntropyLoss', 'BCEWithLogitsLoss', 'MSELoss', 'L1Loss'。
        **kwargs: 傳遞給損失函數構造器的額外參數。

    Returns:
        torch.nn.Module: 損失函數實例。

    Raises:
        ValueError: 如果提供不支援的損失函數名稱。
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'crossentropy' or loss_name == 'crossentropyloss':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'bcewithlogits' or loss_name == 'bcewithlogitsloss':
        # BCEWithLogitsLoss 適用於二分類或多標籤分類，直接作用於模型輸出 (logits)，內部包含 Sigmoid
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.25]),**kwargs)
    elif loss_name == 'mse' or loss_name == 'mseloss':
        return nn.MSELoss(**kwargs)
    elif loss_name == 'l1' or loss_name == 'l1loss':
        return nn.L1Loss(**kwargs)
    elif loss_name == 'wce':
        return nn.CrossEntropyLoss(weight=torch.FloatTensor([0.2, 0.8]))
    else:
        raise ValueError(f"不支援的損失函數: {loss_name}。請選擇 'CrossEntropyLoss', 'BCEWithLogitsLoss', 'MSELoss', 'L1Loss'。")

# 範例使用 (假設你的配置中有 loss_name 參數)
# loss_fn = get_loss_function(config.train.loss_name, weight=torch.tensor([1.0, 2.0])) # 帶權重的交叉熵
# loss_fn_bce = get_loss_function('BCEWithLogitsLoss', pos_weight=torch.tensor([10.0])) # 帶正樣本權重的 BCE

if __name__ == "__main__":
    print(get_loss_function("crossentropy"))