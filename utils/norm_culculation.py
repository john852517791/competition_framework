import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def calculate_dataset_normalization_metrics(label_file_path, image_folder_path):
    """
    计算给定数据集的归一化指标（均值和标准差）。

    Args:
        label_file_path (str): 标签文件的路径，每行包含 '文件名 标签名'。
        image_folder_path (str): 图像文件所在的根目录路径。

    Returns:
        tuple: 包含均值和标准差的元组。
               例如：(mean=[R_mean, G_mean, B_mean], std=[R_std, G_std, B_std])
    """

    # 用于将图像转换为张量并调整大小（可选，但推荐统一处理）
    # 这里我们只做To_Tensor，不做Normalize，因为Normalize是我们要计算的目标
    transform_to_tensor = transforms.ToTensor()

    # 存储每个通道的像素值总和
    sum_pixels = torch.tensor([0.0, 0.0, 0.0])
    # 存储每个通道的像素值平方和（用于计算标准差）
    sum_squared_pixels = torch.tensor([0.0, 0.0, 0.0])
    # 存储所有图像的像素总数
    num_pixels = 0

    print(f"开始计算数据集的归一化指标...")

    with open(label_file_path, 'r') as f:
        lines = f.readlines()

    # 遍历所有图像
    for i, line in enumerate(lines):
        try:
            parts = line.strip().split()
            if not parts:  # 跳过空行
                continue
            image_filename = parts[0]
            # 标签名如果不需要计算可以忽略 parts[1]

            image_path = os.path.join(image_folder_path, image_filename)

            if not os.path.exists(image_path):
                print(f"警告：文件不存在，跳过：{image_path}")
                continue

            img = Image.open(image_path).convert('RGB') # 确保图像是RGB格式
            tensor_img = transform_to_tensor(img) # 转换为张量，像素值范围 [0, 1]

            # 图像形状是 (C, H, W)，所以我们计算每个通道的均值和标准差
            # 累加每个通道的像素值总和
            sum_pixels += tensor_img.sum(dim=[1, 2])
            # 累加每个通道的像素值平方和
            sum_squared_pixels += (tensor_img ** 2).sum(dim=[1, 2])

            # 累加像素总数
            num_pixels += tensor_img.shape[1] * tensor_img.shape[2]

            if (i + 1) % 10000 == 0:
                print(f"已处理 {i + 1}/{len(lines)} 张图像...")

        except Exception as e:
            print(f"处理文件 {image_filename} 时发生错误: {e}")
            continue

    if num_pixels == 0:
        raise ValueError("数据集中没有找到有效的图像或像素。请检查文件路径和标签文件。")

    # 计算均值
    mean = sum_pixels / num_pixels

    # 计算标准差
    # 方差 = (像素平方和 / 像素总数) - 均值平方
    variance = (sum_squared_pixels / num_pixels) - (mean ** 2)
    # 确保方差非负，避免浮点数误差导致负值
    std = torch.sqrt(torch.clamp(variance, min=1e-7))

    # 将结果转换为列表，方便后续使用
    mean_list = mean.tolist()
    std_list = std.tolist()

    print("\n计算完成！")
    print(f"数据集的均值: {mean_list}")
    print(f"数据集的标准差: {std_list}")

    return mean_list, std_list

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设您的标签文件名为 'labels.txt'
    # 假设您的图像都存放在 'images/' 文件夹下
    # 请根据您的实际情况修改以下路径
    current_dir = "./"
    example_label_file = os.path.join(current_dir, 'merged_deduplicated.txt')
    example_image_folder = "data/train/images"

    try:
        calculated_mean, calculated_std = calculate_dataset_normalization_metrics(
            label_file_path=example_label_file,
            image_folder_path=example_image_folder
        )

        print("\n您可以使用以下 `transforms.Normalize` 配置：")
        print(f"transforms.Normalize(mean={calculated_mean}, std={calculated_std})")

    except ValueError as e:
        print(f"错误：{e}")
    except FileNotFoundError:
        print(f"错误：请检查标签文件路径 '{example_label_file}' 或图像文件夹路径 '{example_image_folder}' 是否正确。")
    except Exception as e:
        print(f"发生未知错误: {e}")