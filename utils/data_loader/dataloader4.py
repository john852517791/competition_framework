import sys
sys.path.append("./")
import torch
from torch.utils.data import Dataset, DataLoader,default_collate
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import lightning as pl
from config.config_class import DataConfig,AugmentationConfig
import random
from torchvision.transforms import v2
import cv2,yaml
import albumentations as A
import numpy as np
from utils.data_loader.albu import IsotropicResize

# MEAN = [0.4972752332687378, 0.4161258339881897, 0.38086166977882385]
# STD = [0.3028480112552643, 0.2794816493988037, 0.27611133456230164]
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

def read_yml():
    with open("config/data_aug.yaml", 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    return config

def init_data_aug_method(config = read_yml()):
    trans = A.Compose([           
        A.HorizontalFlip(p=config['data_aug']['flip_prob']),
        A.Rotate(limit=config['data_aug']['rotate_limit'], p=config['data_aug']['rotate_prob']),
        A.GaussianBlur(blur_limit=config['data_aug']['blur_limit'], p=config['data_aug']['blur_prob']),
        A.OneOf([                
            IsotropicResize(max_side=config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p = 0 if config['with_landmark'] else 1),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=config['data_aug']['brightness_limit'], contrast_limit=config['data_aug']['contrast_limit']),
            A.FancyPCA(),
            A.HueSaturationValue()
        ], p=0.5),
        A.ImageCompression(quality_range = (config['data_aug']['quality_lower'],config['data_aug']['quality_upper']), p=0.5)
    ], 
        keypoint_params=A.KeypointParams(format='xy') if config['with_landmark'] else None
    )
    return trans


class JpgImageDataset(Dataset):
    def __init__(self, folder_path):
        """
        Args:
            folder_path (string): Path to the folder containing JPG images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        self.transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD) # Example ImageNet normalization
    ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.folder_path, img_name)
        
        # Open image using PIL
        image = Image.open(img_path).convert('RGB') # Ensure image is in RGB format

        if self.transform:
            image = self.transform(image)

        return image, img_name




class CustomSharpen:
    """Custom Sharpen transform for PIL Images."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.SHARPEN)
        return img

class CustomGaussianNoise:
    """Custom Gaussian Noise transform for PyTorch Tensors."""
    def __init__(self, mean=0.0, std=0.01, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img_tensor):
        if random.random() < self.p:
            noise = torch.randn(img_tensor.size()) * self.std + self.mean
            return img_tensor + noise
        return img_tensor


class ImageFolderWithTxtLabelsDataset(Dataset):
    """
    一个自定义的PyTorch数据集，用于从文件夹加载图片，并从TXT文件加载对应的标签。
    TXT文件的格式应为：文件名 标签
    """
    def __init__(self, image_folder, label_file_path, transform=None):
        self.image_folder = image_folder
        self.label_file_path = label_file_path
        self.norm = transform["norm"]
        self.aug_t = transform["aug"]
        self.image_labels = [] # 存储 (image_path, label) 对

        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"标签文件未找到: {label_file_path}")

        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, label_str = parts[0], parts[1]
                    try:
                        # label = float(label_str) # 尝试将标签转换为整数
                        label = int(label_str) # 尝试将标签转换为整数
                    except ValueError:
                        print(f"警告: 标签 '{label_str}' 无法转换为整数，已跳过文件 '{filename}'。")
                        continue

                    image_path = os.path.join(self.image_folder, filename)
                    if os.path.exists(image_path) and image_path.lower().endswith(('.jpg', '.jpeg')):
                        self.image_labels.append((image_path, label))
                    else:
                        print(f"警告: 文件 '{filename}' ({image_path}) 未找到或不是JPG/JPEG图片，已跳过。")
                else:
                    print(f"警告: 标签文件行格式不正确，已跳过: {line.strip()}")

        if not self.image_labels:
            raise RuntimeError(f"在 '{image_folder}' 和 '{label_file_path}' 中没有找到有效的图片-标签对。")
        print(f"Dataset 初始化完成：共找到 {len(self.image_labels)} 个有效图片-标签对。")

    def __len__(self):
        return len(self.image_labels)
    
    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = 224 # if self.mode == "train" else self.config['resolution']
        assert os.path.exists(file_path), f"{file_path} does not exist"
        img = cv2.imread(file_path) #bgr格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.image_labels[idx]

        # image = Image.open(image_path).convert('RGB')
        image = self.load_rgb(image_path)
        image = np.array(image)
        if self.aug_t:
            image = self.aug_t(image=image)['image']
        if self.norm:
            image = self.norm(image)

        return image, label


## 改造后的 Lightning DataModule (从 args 获取参数)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, args:DataConfig, aug_args:AugmentationConfig):
        """
        Args:
            args: 一个包含所有配置参数的命名空间对象 (例如，由 argparse.parse_args() 返回)。
                  预期包含以下属性:
                  - args.image_folder (str)
                  - args.label_file_path (str)
                  - args.batch_size (int)
                  - args.num_workers (int)
                  - args.image_size (int 或 tuple)
        """
        super().__init__()
        # 从 args 对象中获取所有必要的参数
        # self.args = args
        self.image_folder = args.image_folder
        self.label_file_path = args.label_file_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.mixup_transform = v2.MixUp(num_classes=None)
        self.mixup_probability = 0.3

        # 检查 image_size 是否为整数，如果是，则转换为 (size, size) 元组
        if isinstance(args.image_size, int):
            self.image_size = (args.image_size, args.image_size)
        elif isinstance(args.image_size, (list, tuple)) and len(args.image_size) == 2:
            self.image_size = tuple(args.image_size)
        else:
            raise ValueError("args.image_size 必须是整数或包含两个整数的元组/列表。")

        self.aug_args = aug_args
        self.aug_transform = init_data_aug_method()

        self.transform = {
            "norm":transforms.Compose(
                [
                    transforms.ToTensor(), # Converts a PIL Image or numpy.ndarray (HxWxC) to a PyTorch FloatTensor (CxHxW) and scales pixel values to [0.0, 1.0].
                    transforms.Normalize(mean=MEAN, std=STD) # Normalizes a tensor image with mean and standard deviation.
                ]),
            "aug":self.aug_transform
            }
    def prepare_data(self):
        # 对于我们当前的情况，数据（图片和标签文件）假设已经存在于本地。
        pass

    def setup(self, stage: str):
        # 实际项目中，你可能需要根据标签文件中的逻辑，或者使用额外的文件来区分训练、验证和测试集
        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolderWithTxtLabelsDataset(self.image_folder, self.label_file_path, self.transform)
            # 假设验证集与训练集相同，实际应用中应有单独的验证集数据或从训练数据中划分
            self.val_dataset = ImageFolderWithTxtLabelsDataset(self.image_folder, self.label_file_path, self.transform)
        if stage == "test" or stage is None:
            self.test_dataset = ImageFolderWithTxtLabelsDataset(self.image_folder, self.label_file_path, self.transform)
        if stage == "predict" or stage is None:
            self.predict_dataset = JpgImageDataset(folder_path="data/test1/images/")

    
    def collate_fn_mixup(self, batch):
        ## default_collate 将单个样本列表转换为批次张量
        images, labels = default_collate(batch)
        # 确保标签是浮点型且形状为 [batch_size, 1]，以匹配 BCEWithLogitsLoss 和 Mixup 的要求
        # 这一步在 Mixup 之前进行，确保标签格式正确
        # if labels.dim() == 1:
            # labels = labels.float().unsqueeze(1) # 从 [64] 变为 [64, 1]
        # --- 引入随机概率判断 ---
        # if self.mixup_transform and random.random() < self.mixup_probability:
        #     # 如果随机数小于设定概率，则应用 Mixup
        #     images_mixed, labels_mixed = self.mixup_transform(images, labels)
        #     return images_mixed, labels_mixed
        # else:
        #     # 否则，返回原始批次
        return images, labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          collate_fn=self.collate_fn_mixup
                          )

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset,
    #                       batch_size=self.batch_size,
    #                       shuffle=False,
    #                       num_workers=self.num_workers,
    #                       pin_memory=True)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset,
    #                       batch_size=self.batch_size,
    #                       shuffle=False,
    #                       num_workers=self.num_workers,
    #                       pin_memory=True)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True)

if __name__ == "__main__":
    from config.config_class import DataConfig,AugmentationConfig
    dm = MyDataModule(args=DataConfig(),aug_args=AugmentationConfig())
    dm.setup("fit")
    trn_dl = dm.train_dataloader()
    print(len(trn_dl))
    for ele in trn_dl:
        print(ele)
        feature, label = ele
        print(feature.shape)
        print(label.shape)
        break