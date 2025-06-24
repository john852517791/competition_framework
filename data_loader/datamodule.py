# pl_project/datamodule.py

import os
import glob
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import lightning as l

class ImageClassificationDataset(Dataset):
    """用于图像分类的自定义数据集。"""
    def __init__(self, image_paths: List[str], class_to_idx: Dict[str, int], processor):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # 从路径中提取类别名称 (例如: /path/to/data/dog/image.jpg -> dog)
        class_name = os.path.basename(os.path.dirname(image_path))
        label = self.class_to_idx[class_name]

        # 处理器负责图像的缩放、归一化和张量转换
        inputs = self.processor(images=image, return_tensors="pt")
        # 移除处理器添加的批次维度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        
        return inputs

class ImageDataModule(l.LightningDataModule):
    """用于图像数据的 LightningDataModule。"""
    def __init__(self, data_dir: str, processor, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters(ignore=['processor'])
        self.data_dir = data_dir
        self.processor = processor
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_to_idx = {}

    def setup(self, stage: Optional[str] = None):
        """在每个 GPU 上执行，用于数据切分和数据集创建。"""
        if not self.train_dataset and not self.val_dataset:
            class_dirs = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
            if not class_dirs:
                raise ValueError(f"在 {self.data_dir} 中未找到类别子目录。期望的结构: data_dir/class_name/images.jpg")
            
            self.class_to_idx = {class_name: i for i, class_name in enumerate(class_dirs)}
            
            all_image_paths = glob.glob(os.path.join(self.data_dir, "**", "*.jpg"), recursive=True)
            if not all_image_paths:
                raise ValueError(f"在 {self.data_dir} 的子目录中未找到 JPG 图像。")

            full_dataset = ImageClassificationDataset(all_image_paths, self.class_to_idx, self.processor)

            # 切分数据集
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
            )
            self.test_dataset = self.val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, pin_memory=True)

