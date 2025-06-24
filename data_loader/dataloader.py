# pl_project/model_loader.py

from transformers import AutoModel, AutoImageProcessor, AutoConfig

def load_model_and_processor(model_name_or_path: str):
    """
    从 Hugging Face Hub 加载模型结构及其对应的图像处理器。
    注意：此版本不加载预训练权重，模型权重是随机初始化的。

    Args:
        model_name_or_path (str): 模型的名称或本地路径。
    """
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModel.from_config(config)
    processor = AutoImageProcessor.from_pretrained(model_name_or_path)
    print(f"成功从配置初始化模型 (不加载预训练权重): '{model_name_or_path}'")
    return model, processor