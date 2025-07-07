import sys
sys.path.append("./")
import torch
import torch.nn as nn
import torch.nn.functional as F
from tl_models.model_utils.efficienctnet import EfficientNet

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.net = EfficientNet.from_name(
            model_name="efficientnet-b6"
            )
        # self.net = EfficientNet()
    def forward(self, x,train = True):
        if train:
            hid,logits = self.net(x)
        else:
            hid,logits
        return hid,logits
    
    
if __name__ == "__main__":
    model = Model(args=None)
    print("--- 模型結構 ---")
    print(model)
    dummy_input = torch.randn(4, 3, 384, 384) 
    model.eval() 
    # 進行前向傳播
    with torch.no_grad(): # 在推理時不需要計算梯度
        _, logits = model(dummy_input)
    print("\n--- 輸出形狀 ---")
    print(f"輸入圖像形狀: {dummy_input.shape}")
    print(f"輸入圖像形狀: {logits.shape}")