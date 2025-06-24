# PyTorch Lightning 项目模板

本项目提供了一个结构化的模板，用于使用 PyTorch Lightning 在图像分类任务上训练模型（例如 Vision Transformer）。代码被模块化为：

- `main.py`: 运行训练的主脚本。
- `model_loader.py`: 处理从 Hugging Face 加载预训练模型。
- `datamodule.py`: 用于加载和准备图像数据集的 `LightningDataModule`。
- `tl_model.py`: 包装模型并定义训练、验证和优化逻辑的 `LightningModule`。

## 环境设置

### 1. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate
```

### 2. 安装依赖

```bash
pip install torch torchvision pytorch-lightning transformers Pillow pyyaml
```

### 3. 准备数据集

您的数据应按以下结构组织，其中 `path/to/your/data` 中的每个子目录代表一个类别：

```
path/to/your/data/
├── class_a/
│   ├── image_001.jpg
│   └── ...
├── class_b/
│   ├── image_101.jpg
│   └── ...
└── ...
```

## 如何运行

通过运行 `main.py` 脚本并提供必要的参数来开始训练。最重要的参数是 `--data_dir`，它应该指向您的数据集目录。

```bash
python main.py --data_dir /path/to/your/data --batch_size 32 --max_epochs 20 --accelerator gpu --devices 1
```

训练进度、日志和最佳模型 checkpoint 将保存在 `--output_dir` 中。