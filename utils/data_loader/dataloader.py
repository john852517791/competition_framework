import sys
sys.path.append("./")
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import os
import lightning as pl
from config.config_class import DataConfig,AugmentationConfig
import random


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
        self.transform = transform
        self.image_labels = [] # 存储 (image_path, label) 对

        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"标签文件未找到: {label_file_path}")

        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, label_str = parts[0], parts[1]
                    try:
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

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label = self.image_labels[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

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

        # 检查 image_size 是否为整数，如果是，则转换为 (size, size) 元组
        if isinstance(args.image_size, int):
            self.image_size = (args.image_size, args.image_size)
        elif isinstance(args.image_size, (list, tuple)) and len(args.image_size) == 2:
            self.image_size = tuple(args.image_size)
        else:
            raise ValueError("args.image_size 必须是整数或包含两个整数的元组/列表。")


        # 定义数据转换
        # self.transform = transforms.Compose([
        #     transforms.Resize(self.image_size),
        #     transforms.RandomHorizontalFlip(p=0.5), # 以50%的概率水平翻转
        #     transforms.RandomVerticalFlip(p=0.5), # 以30%的概率垂直翻转
        #     transforms.RandomRotation(degrees=15, fill=0), # 在-15到15度之间随机旋转，边缘填充黑色
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.aug_args = aug_args
        transform_list = [
            transforms.Resize(self.image_size),
        ]

        # 根据配置添加数据增强
        if self.aug_args.random_horizontal_flip_p > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=self.aug_args.random_horizontal_flip_p))
        
        if self.aug_args.random_vertical_flip_p > 0:
            transform_list.append(transforms.RandomVerticalFlip(p=self.aug_args.random_vertical_flip_p))
        
        if self.aug_args.random_rotation_degrees > 0:
            transform_list.append(transforms.RandomRotation(degrees=self.aug_args.random_rotation_degrees, fill=0))
        
        # ColorJitter只有当至少一个参数大于0时才添加
        if any([self.aug_args.color_jitter_brightness > 0, 
                self.aug_args.color_jitter_contrast > 0, 
                self.aug_args.color_jitter_saturation > 0, 
                self.aug_args.color_jitter_hue > 0]):
            transform_list.append(transforms.ColorJitter(
                brightness=self.aug_args.color_jitter_brightness,
                contrast=self.aug_args.color_jitter_contrast,
                saturation=self.aug_args.color_jitter_saturation,
                hue=self.aug_args.color_jitter_hue
            ))

        if self.aug_args.gaussian_blur_p > 0:
            # kernel_size的选择通常取决于图像大小，这里假设一个通用值
            transform_list.append(transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)))

        # 先ToTensor，因为Sharpen和Gaussian Noise可能需要对Tensor操作
        transform_list.append(transforms.ToTensor())

        if self.aug_args.sharpen_p > 0:
            # 这里的 CustomSharpen 假设它在 ToTensor 之前作用于 PIL Image
            # 为了能在 ToTensor 之后使用，你需要重写 CustomSharpen 或使用其他库 (如 Kornia)
            print("Warning: Sharpening is applied before ToTensor to use PIL's ImageFilter.SHARPEN. "
                  "If you need tensor-based sharpening, consider using Kornia or custom implementation.")
            pass # 实际 CustomSharpen 应该在 ToTensor 之前，或者你需要自定义一个对 Tensor 锐化的操作。
        
        if self.aug_args.add_gaussian_noise_p > 0:
            # 高斯噪声通常在 ToTensor 之后对 Tensor 进行操作
            transform_list.append(CustomGaussianNoise(
                mean=self.aug_args.add_gaussian_noise_mean,
                std=self.aug_args.add_gaussian_noise_std,
                p=self.aug_args.add_gaussian_noise_p
            ))
        
        # RandomErasing 必须在 ToTensor 之后，因为它操作的是 Tensor
        if self.aug_args.random_erasing_p > 0:
            transform_list.append(transforms.RandomErasing(p=self.aug_args.random_erasing_p))

        transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = transforms.Compose(transform_list)

    def prepare_data(self):
        # 对于我们当前的情况，数据（图片和标签文件）假设已经存在于本地。
        pass

    def setup(self, stage: str):
        # 实际项目中，你可能需要根据标签文件中的逻辑，或者使用额外的文件来区分训练、验证和测试集
        if stage == "fit" or stage is None:
            self.train_dataset = ImageFolderWithTxtLabelsDataset(self.image_folder, self.label_file_path, self.transform)
            # 假设验证集与训练集相同，实际应用中应有单独的验证集数据或从训练数据中划分
            # self.val_dataset = ImageFolderWithTxtLabelsDataset(self.image_folder, self.label_file_path, self.transform)
        if stage == "test" or stage is None:
            self.test_dataset = ImageFolderWithTxtLabelsDataset(self.image_folder, self.label_file_path, self.transform)
        if stage == "predict" or stage is None:
            self.predict_dataset = ImageFolderWithTxtLabelsDataset(self.image_folder, self.label_file_path, self.transform)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True)

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

    # def predict_dataloader(self):
    #     return DataLoader(self.predict_dataset,
    #                       batch_size=self.batch_size,
    #                       shuffle=False,
    #                       num_workers=self.num_workers,
    #                       pin_memory=True)

if __name__ == "__main__":
    from config.config_class import DataConfig,AugmentationConfig
    dm = MyDataModule(args=DataConfig(),aug_args=AugmentationConfig())
    dm.setup("fit")
    trn_dl = dm.train_dataloader()
    for ele in trn_dl:
        print(ele)
        feature, label = ele
        print(feature.shape)
        print(label.shape)
        break