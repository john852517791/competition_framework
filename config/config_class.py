from dataclasses import dataclass, field, asdict
import torch
import yaml
import os

@dataclass
class DataConfig:
    image_folder: str = 'data/train/images'
    label_file_path: str = 'data/train.txt'
    # Define how to split your dataset
    # train_ratio: float = 0.8
    # val_ratio: float = 0.1
    # test_ratio: float = 0.1 # Ensure train_ratio + val_ratio + test_ratio = 1.0
    image_size: int = 224 # Unified image size
    num_classes: int = 2 # Assuming binary classification: real vs. fake
    batch_size: int = 32
    num_workers: int = 8

@dataclass
class TrainConfig:
    seed:int = 18181
    sample_num:int = 5302480
    epochs: int = 5
    gradient_clip_val: float = 1.5
    gradient_clip_algorithm: str = "norm"
    # Learning rate scheduler
    loss_fn:str = "bcewithlogitsloss"
    learning_rate: float = 5e-4
    optimizer: str = 'AdamW' # 'Adam', 'SGD', 'AdamW'
    weight_decay: float = 1e-3
    momentum: float = 0.9 # For SGD
    patience: int = 3
    save_top_k: int = -1
    accelerator: str = "gpu"
    deviceid: list[int] = field(default_factory=lambda: [0,1,2,3,4,5,6,7])
    batch_size = DataConfig.batch_size
    
    lr_scheduler: str = 'cosineannealinglr' # 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'None', 'LinearWarmupCosineAnnealingLR'
    lr_scheduler_step_size: int = 5 # For StepLR
    lr_scheduler_gamma: float = 0.1 # For StepLR
    lr_scheduler_mode: str = 'min' # For ReduceLROnPlateau ('min' or 'max')
    lr_scheduler_patience: int = 3 # For ReduceLROnPlateau
    lr_scheduler_factor: float = 0.1 # For ReduceLROnPlateau
    # lr_scheduler_t_max: int = 10 # For CosineAnnealingLR (total epochs or steps)
    # Precision for training
    # precision: str = '32-true' # '32-true', '16-mixed', 'bf16-mixed'
    # Warmup parameters
    warmup_steps: int = 4000 # Number of steps/epochs for warmup. If 0, no warmup.

@dataclass
class AugmentationConfig:
    """Data augmentation configurations for image processing."""
    random_horizontal_flip_p: float = 0.5
    random_vertical_flip_p: float = 0.0
    random_rotation_degrees: int = 15
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    random_erasing_p: float = 0.0
    gaussian_blur_p: float = 0.0
    # Add more specific augmentations for fake image detection
    # Consider augmentations that mimic common fake artifacts or enhance subtle details
    sharpen_p: float = 0.0 # Probability of applying sharpening
    add_gaussian_noise_p: float = 0.0 # Probability of adding Gaussian noise
    add_gaussian_noise_mean: float = 0.0
    add_gaussian_noise_std: float = 0.01


@dataclass
class moduleConfig:
    """Model specific configurations."""
    dataModule: str = "utils/data_loader/dataloader2.py"
    modelModule: str = "tl_models/models/efficientnet-b4_1.py"
    tl_model_module: str = "tl_models/tl_model.py"
    
@dataclass
class LoggingConfig:
    """Configurations for logging with TensorBoard."""
    log_dir: str = './a_logs'
    single_logname:str = "efficientnet-b4_1"
    experiment_name: str = 'fake_image_detection'
    # log_every_n_steps: int = 50 # How often to log metrics to TensorBoard

@dataclass
class GlobalConfig:
    
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentationConfig = field(default_factory=AugmentationConfig)
    module: moduleConfig = field(default_factory=moduleConfig) # New module config
    logging: LoggingConfig = field(default_factory=LoggingConfig) # New logging config

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """
        Loads configuration from a YAML file and creates a GlobalConfig instance.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        # Initialize nested dataclasses, applying defaults if keys are missing
        data_config = DataConfig(**yaml_data.get('data', {}))
        train_config = TrainConfig(**yaml_data.get('train', {}))
        augment_config = AugmentationConfig(**yaml_data.get('augment', {}))
        module_config = moduleConfig(**yaml_data.get('module', {}))
        logging_config = LoggingConfig(**yaml_data.get('logging', {}))

        # Handle device specific logic for train_config if not explicitly in YAML
        # if 'device' not in yaml_data.get('train', {}):
        #     train_config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        return cls(data=data_config, train=train_config, augment=augment_config,
                   module=module_config, logging=logging_config)

    def to_yaml(self, yaml_path: str):
        """
        Saves the current GlobalConfig instance to a YAML file.
        """
        config_dict = asdict(self)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        print(f"Configuration saved to {yaml_path}")

if __name__ == "__main__":
    print("--- Loading config from sample.yaml ---")
    ymlfile = "config/config.yaml"
    gb_conf = GlobalConfig()
    print(gb_conf)
    print("\n--- Saving config back to sample.yaml ---")
    gb_conf.to_yaml(ymlfile) # Save it back to verify

    # Example of creating a default config and saving it
    print("\n--- Creating and saving default config ---")
    gb_conf = GlobalConfig.from_yaml(ymlfile)
    print(gb_conf)
    print(gb_conf.train.loss_fn)