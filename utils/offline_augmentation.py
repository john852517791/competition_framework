import os
import cv2
import numpy as np
import albumentations as A
import random
import sys
from tqdm import tqdm # 用于显示进度条

def augment_data_by_methods(
    id1,
    data_folder: str,
    label_file_name: str,
    output_label_file_name: str = "all_augmented_labels.txt",
    output_image_folder: str = None,
    image_size: int = 384,
    seed: int = 42, # 主随机种子，用于控制所有增强过程的可复现性
    flip_prob: float = 1.0, # 水平翻转的概率，这里设为1.0以确保所有图像都被翻转
    gaussian_blur_limit: int = 7, # 模糊核的最大尺寸，必须是奇数
    brightness_limit: float = 0.2, # 亮度调整范围
    contrast_limit: float = 0.2, # 对比度调整范围
    fancy_pca_alpha: float = 0.1, # FancyPCA 强度参数
    hsv_shift_limits: tuple = (20, 30, 20), # (hue, sat, val) 调整范围
    quality_lower: int = 75, # 图像压缩质量下限
    quality_upper: int = 95, # 图像压缩质量上限
):
    """
    对指定文件夹下的JPG图像，按每个方法独立进行一倍的数据增强，并更新标签文件。

    Args:
        data_folder (str): 包含JPG图像和标签文件的文件夹路径。
        label_file_name (str): 原始标签文件的名称 (例如 "labels.txt")。
        output_label_file_name (str): 输出标签文件的名称 (例如 "all_augmented_labels.txt")。
        output_image_folder (str, optional): 增强图像的保存路径。建议指定一个不同的文件夹。
        image_size (int): 图像的目标尺寸（宽度和高度）。
        seed (int): 用于所有随机操作的主随机种子，确保可复现性。
        flip_prob (float): 水平翻转的概率。设置为1.0以确保每张图都被翻转。
        gaussian_blur_limit (int): 高斯模糊核的最大尺寸。必须是奇数。
        brightness_limit (float): RandomBrightnessContrast的亮度限制。
        contrast_limit (float): RandomBrightnessContrast的对比度限制。
        fancy_pca_alpha (float): FancyPCA的强度参数。
        hsv_shift_limits (tuple): HueSaturationValue的(hue, sat, val)调整范围。
        quality_lower (int): ImageCompression的质量下限。
        quality_upper (int): ImageCompression的质量上限。
    """

    # --- 1. 设置主随机种子，确保整个过程的可复现性 ---
    random.seed(seed)
    np.random.seed(seed)
    # Albumentations 内部会使用 numpy.random，所以设置 np.random.seed 即可

    # --- 2. 准备文件路径和输出文件夹 ---
    original_label_path = label_file_name
    output_label_path = output_label_file_name
    
    if output_image_folder:
        os.makedirs(output_image_folder, exist_ok=True)
    else:
        output_image_folder = data_folder # 如果未指定输出文件夹，则使用原始数据文件夹

    # 检查原始标签文件是否存在
    if not os.path.exists(original_label_path):
        print(f"错误：原始标签文件 '{original_label_path}' 不存在。")
        return

    # --- 3. 定义每种独立的数据增强流水线 ---
    # 注意：这里我们让每个增强方法独立工作，所以它们的概率设为1.0
    # 模糊核尺寸限制必须是奇数
    blur_ksize_limit = (3, gaussian_blur_limit) if gaussian_blur_limit >= 3 and gaussian_blur_limit % 2 != 0 else (3, 7)
    if gaussian_blur_limit % 2 == 0:
        print(f"警告: gaussian_blur_limit ({gaussian_blur_limit}) 必须是奇数，已调整为 {blur_ksize_limit[1]}")

    augmentations = [
        ["_flip", A.HorizontalFlip(p=flip_prob)],
        ["_blur", A.GaussianBlur(blur_limit=blur_ksize_limit, p=1.0)], # 1.0表示每次都应用
        ["_brightness_contrast", A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=1.0)],
        ["_fancy_pca", A.FancyPCA(alpha=fancy_pca_alpha, p=1.0)],
        ["_hsv", A.HueSaturationValue(hue_shift_limit=hsv_shift_limits[0], sat_shift_limit=hsv_shift_limits[1], val_shift_limit=hsv_shift_limits[2], p=1.0)],
        ["_compress_JPEG", A.ImageCompression(quality_range=(quality_lower,quality_upper),compression_type='jpeg', p=1.0)],
        ["_compress_WebP", A.ImageCompression(quality_range=(quality_lower,quality_upper),compression_type='webp', p=1.0)],
    ]

    # --- 4. 读取原始标签文件 ---
    original_data = []
    try:
        with open(original_label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_name, label = parts
                    original_data.append((image_name, label))
                else:
                    print(f"警告：标签文件 '{original_label_path}' 中存在格式不正确的行：'{line.strip()}'，已跳过。")
    except Exception as e:
        print(f"读取原始标签文件 '{original_label_path}' 时发生错误：{e}")
        return

    # --- 5. 初始化新的标签列表（包含原始数据） ---
    all_augmented_labels_lines = []
    for img_name, label in original_data:
        all_augmented_labels_lines.append(f"{img_name} {label}")

    print(f"开始处理 {len(original_data)} 张原始图像，并应用每种增强方法...")
    
    # --- 6. 遍历原始数据，依次应用每种增强方法 ---
    for original_idx, (image_name, label) in enumerate(tqdm(original_data, desc="应用数据增强方法")):
        original_image_path = os.path.join(data_folder, image_name)

        if not os.path.exists(original_image_path):
            print(f"警告：图像文件 '{original_image_path}' 不存在，已跳过。")
            continue

        try:
            image = cv2.imread(original_image_path)
            if image is None:
                print(f"警告：无法读取图像 '{original_image_path}'，已跳过。")
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 确保图像尺寸符合预期
            if image.shape[0] != image_size or image.shape[1] != image_size:
                image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)

            # 为每个增强方法创建副本，并应用
            for [aug_suffix, aug_transform] in augmentations[id1:(id1+1)]:
                # print(aug_suffix)
                # 为了可复现性，每次调用 transform 都依赖于全局设置的随机种子
                # 这样，即使是像高斯模糊这样的随机参数选择，也会是可预测的。
                augmented_image = aug_transform(image=image)['image']

                augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

                # 生成新的文件名
                base_name = os.path.basename(image_name)
                original_base, original_ext = os.path.splitext(base_name)
                new_image_name = f"{original_base}{aug_suffix}{original_ext}"
                new_image_path = os.path.join(output_image_folder, new_image_name)

                # 保存增强后的图像
                cv2.imwrite(new_image_path, augmented_image_bgr)

                # 添加到新的标签列表
                all_augmented_labels_lines.append(f"{new_image_name} {label}")

        except Exception as e:
            print(f"处理图像 '{original_image_path}' 时发生错误：{e}")
            print(f"错误信息: {e}")
            sys.exit(1) # 遇到错误时退出，确保数据一致性

    # --- 7. 保存新的标签文件 ---
    try:
        with open(output_label_path, 'w', encoding='utf-8') as f:
            for line in all_augmented_labels_lines:
                f.write(line + '\n')
        print(f"\n数据增强完成！")
        print(f"增强后的图像保存在：{output_image_folder}")
        print(f"新的标签文件保存在：{output_label_path}")
        print(f"总共处理了 {len(original_data)} 张原始图像。")
        print(f"应用了 {len(augmentations)} 种增强方法，总计生成了 {len(augmentations) * len(original_data)} 张增强图像。")
        print(f"最终 {len(all_augmented_labels_lines)} 条记录在 {output_label_path} 中 (包含原始图像)。")

    except Exception as e:
        print(f"写入新的标签文件 '{output_label_path}' 时发生错误：{e}")
        return

# --- 使用示例 ---
if __name__ == "__main__":
    # 请根据你的实际情况修改以下参数
    id1 = int(sys.argv[1])
    your_data_folder = "data/train/images/" # 你的数据文件夹路径
    your_label_file = "data/train.txt" # 你的原始标签文件名

    # 可以指定一个不同的输出文件夹来存放增强后的图片，强烈建议这样做
    # 如果设置为 None，增强后的图片将和原始图片放在一起
    your_output_image_folder = None
    your_output_label_file = f"data/train_augs_{id1}.txt" # 新的标签文件名

    # 运行数据增强
    augment_data_by_methods(
        id1,
        data_folder=your_data_folder,
        label_file_name=your_label_file,
        output_label_file_name=your_output_label_file,
        output_image_folder=your_output_image_folder,
        image_size=384, # 你的图像尺寸
        seed=42, # 设置一个你喜欢的整数种子，确保每次运行结果可复现
        # 以下参数可以根据你的需求调整
        flip_prob=1.0, # 确保所有图像都被水平翻转
        gaussian_blur_limit=9, # 高斯模糊核的最大尺寸 (奇数)
        brightness_limit=0.25, # 亮度调整幅度
        contrast_limit=0.25, # 对比度调整幅度
        fancy_pca_alpha=0.15, # FancyPCA 强度
        hsv_shift_limits=(25, 40, 25), # (色相，饱和度，值) 调整幅度
        quality_lower=50, # 图像压缩质量下限
        quality_upper=90, # 图像压缩质量上限,
    )