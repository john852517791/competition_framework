import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        num_classes = 2
        # 第一個卷積塊: 384x384 -> 192x192
        # 輸入通道數為3 (RGB圖像)
        # 輸出通道數為32
        # 卷積核大小為3x3, 步幅為1, 填充為1 (保持空間大小不變)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        # BatchNorm 通常放在 Conv2d 之後，激活函數之前
        self.bn1 = nn.BatchNorm2d(32)
        # 最大池化層: 192x192 -> 96x96 (將空間維度減半)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # 第二個卷積塊: 96x96 -> 48x48
        # 輸入通道數為32
        # 輸出通道數為64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三個卷積塊: 48x48 -> 24x24
        # 輸入通道數為64
        # 輸出通道數為128
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第四個卷積塊: 24x24 -> 12x12
        # 輸入通道數為128
        # 輸出通道數為256
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第五個卷積塊: 12x12 -> 6x6
        # 輸入通道數為256
        # 輸出通道數為512
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # 512 * 6 * 6 特徵圖

        # 全連接層 (分類頭)
        # 經過五次池化後，原始 384x384 圖像尺寸會縮小 2^5 = 32 倍
        # 所以最終特徵圖的空間大小是 384 / 32 = 12
        # 每個通道的大小是 12x12
        # 輸出通道數為 512
        # 因此，展平後的特徵數量是 512 * 12 * 12 = 73728
        # 更正: 384/2 = 192, 192/2 = 96, 96/2 = 48, 48/2 = 24, 24/2 = 12. 
        # 所以是 512 * 12 * 12 = 73728. (我的註釋寫錯了，現在修正為 6x6)
        # 重新計算： 384 -> 192 -> 96 -> 48 -> 24 -> 12
        # 最終特徵圖尺寸應為 12x12。
        # 讓我確認一下池化層後的尺寸。
        # 384x384 --pool1--> 192x192
        # 192x192 --pool2--> 96x96
        # 96x96 --pool3--> 48x48
        # 48x48 --pool4--> 24x24
        # 24x24 --pool5--> 12x12
        # 所以展平後的大小是 512 * 12 * 12 = 73728
        self.fc1 = nn.Linear(512 * 12 * 12, 1024) # 第一個全連接層
        self.dropout = nn.Dropout(0.5) # 防止過擬合
        self.fc2 = nn.Linear(1024, num_classes) # 輸出層，二分類則 num_classes=2

    def forward(self, x,train = True):
        # 輸入 x 的形狀: (batch_size, 3, 384, 384)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # 形狀: (batch_size, 32, 192, 192)
        
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # 形狀: (batch_size, 64, 96, 96)
        
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # 形狀: (batch_size, 128, 48, 48)
        
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        # 形狀: (batch_size, 256, 24, 24)

        features_before_flatten = x # 記錄展平前的特徵
        
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        # 形狀: (batch_size, 512, 12, 12)
        
        # 展平操作，從第二個維度開始展平 (即通道、高度、寬度)
        x = x.view(-1, 512 * 12 * 12)
        # 形狀: (batch_size, 73728)

        # 全連接層
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        # 形狀: (batch_size, num_classes)
        
        return features_before_flatten, logits

# --- 測試模型 ---
if __name__ == "__main__":
    # 創建一個模型實例，用於二分類
    model = Model(None)
    print("--- 模型結構 ---")
    # print(model)

    # 模擬一個輸入圖像 batch
    # batch_size=4, 3個通道 (RGB), 384x384 像素
    dummy_input = torch.randn(4, 3, 384, 384) 

    # 將模型設置為評估模式 (不需要計算梯度，Dropout 等行為會改變)
    model.eval() 

    # 進行前向傳播
    with torch.no_grad(): # 在推理時不需要計算梯度
        features, logits = model(dummy_input)

    print("\n--- 輸出形狀 ---")
    print(f"輸入圖像形狀: {dummy_input.shape}")
    print(f"展平前特徵形狀 (來自 Conv4 之後): {features.shape}") # 應該是 (4, 256, 24, 24)
    print(f"分類頭輸出 (logits) 形狀: {logits.shape}") # 應該是 (4, 2)

    # # 檢查是否在 GPU 上運行 (如果可用)
    # if torch.cuda.is_available():
    #     print(f"\n模型將在 {model.conv1.weight.device} 上運行。")
    #     model.to('cuda')
    #     dummy_input = dummy_input.to('cuda')
    #     with torch.no_grad():
    #         features_gpu, logits_gpu = model(dummy_input)
    #     print(f"GPU 上展平前特徵形狀: {features_gpu.shape}")
    #     print(f"GPU 上分類頭輸出形狀: {logits_gpu.shape}")
    # else:
    #     print("\nCUDA 不可用，模型將在 CPU 上運行。")

    # print("\n--- 模型參數總數 ---")
    # num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"可訓練參數總數: {num_params}")