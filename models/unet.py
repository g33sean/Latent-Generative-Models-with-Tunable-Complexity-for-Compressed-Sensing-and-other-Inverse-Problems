import yaml
from pathlib import Path
from diffusers import UNet2DModel
from typing import Dict, Any

class UNET_MODELS:
    """Factory class for creating UNet2DModel models based on configuration.
    Supports CelebA, MNIST, CIFAR-10, FFHQ, and ImageNet datasets.
    """
    def __init__(self):
        self.dataset_configs = {
            'CELEBA':   self._get_celeba_config,
            'MNIST':    self._get_mnist_config,
            'CIFAR':    self._get_cifar10_config,
            'FFHQ':     self._get_ffhq_config,
            'IMAGENET': self._get_img_config,
        }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_img_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get ImageNet UNet configuration."""
        latent_channels = dataset_params.get('latent_channels', 12)
        num_classes = dataset_params.get('num_classes', 1000)
        return {
            'in_channels':        latent_channels,
            'out_channels':       latent_channels,
            'num_class_embeds':   num_classes + 1,
            'block_out_channels': (320, 320, 640, 640, 1280, 1280),
            'down_block_types': (
                "DownBlock2D",       # 64x64
                "DownBlock2D",       # 32x32
                "AttnDownBlock2D",   # 16x16
                "AttnDownBlock2D",   #  8x8
                "AttnDownBlock2D",   #  4x4
                "AttnDownBlock2D",   #  2x2
            ),
            'up_block_types': (
                "AttnUpBlock2D",     #  2x2
                "AttnUpBlock2D",     #  4x4
                "AttnUpBlock2D",     #  8x8
                "AttnUpBlock2D",     # 16x16
                "UpBlock2D",         # 32x32
                "UpBlock2D",         # 64x64
            ),
            'layers_per_block':        3,
            'attention_head_dim':      64,
            'norm_num_groups':         32,
            'act_fn':                  "silu",
            'resnet_time_scale_shift': "scale_shift",
        }
    def _get_ffhq_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get FFHQ-specific UNet configuration."""
        latent_channels = dataset_params.get('latent_channels', 24)
        return {
            'in_channels':        latent_channels,
            'out_channels':       latent_channels,
            'block_out_channels': (320, 640, 960, 1280),
            'down_block_types':   ("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            'up_block_types':     ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
            'layers_per_block':   3,
            'attention_head_dim': 64,
            'norm_num_groups':    32,
            'act_fn':             "silu",
            'resnet_time_scale_shift': "scale_shift",

        }

    def _get_celeba_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get CelebA-specific UNet configuration. FID ~6 achieved."""
        latent_channels = dataset_params.get('latent_channels', 12)
        return {
            'in_channels':             latent_channels,
            'out_channels':            latent_channels,
            'block_out_channels':      (256, 512, 768, 1024),
            'down_block_types': (
                "AttnDownBlock2D",     # 64x64
                "AttnDownBlock2D",     # 32x32
                "AttnDownBlock2D",     # 16x16
                "DownBlock2D",         #  8x8
            ),
            'up_block_types': (
                "UpBlock2D",           #  8x8
                "AttnUpBlock2D",       # 16x16
                "AttnUpBlock2D",       # 32x32
                "AttnUpBlock2D",       # 64x64
            ),
            'layers_per_block':        2,
            'attention_head_dim':      64,
            'norm_num_groups':         32,
            'act_fn':                  "silu",
        }

    def _get_mnist_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get MNIST-specific UNet configuration (~1.2M params)."""
        im_channels = dataset_params.get('im_channels', 1)
        return {
            'in_channels':        im_channels,
            'out_channels':       im_channels,
            # 2 levels: 32x32 -> 16x16
            'block_out_channels': (64, 128),
            'down_block_types':   ("AttnDownBlock2D", "AttnDownBlock2D"),
            'up_block_types':     ("AttnUpBlock2D",   "AttnUpBlock2D"),
            'layers_per_block':   1,
            'norm_num_groups':    8,
            'act_fn':             "silu",
        }

    def _get_cifar10_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get CIFAR-10 specific UNet configuration."""
        latent_channels = dataset_params.get('latent_channels', 8)
        return {
            'in_channels':        latent_channels,
            'out_channels':       latent_channels,
            # 3 levels: 8x8 -> 4x4 -> 2x2
            'block_out_channels': (128, 256, 512),
            'down_block_types': (
                "AttnDownBlock2D",     # 8x8  — small enough for attention
                "DownBlock2D",         # 4x4
                "DownBlock2D",         # 2x2
            ),
            'up_block_types': (
                "UpBlock2D",           # 2x2
                "UpBlock2D",           # 4x4
                "AttnUpBlock2D",       # 8x8
            ),
            'layers_per_block':   1,
            'attention_head_dim': 32,
            'norm_num_groups':    32,
            'act_fn':             "silu",
        }

    def create_unet_from_dataset(self, dataset_params: Dict[str, Any]) -> UNet2DModel:
        """Create a UNet2DModel based on dataset name in dataset_params."""
        name = dataset_params.get('name')
        if not name:
            raise ValueError("Please specify dataset_params['name'] (e.g., 'CELEBA', 'MNIST', or 'CIFAR').")
        dataset_name = name.upper()
        if dataset_name not in self.dataset_configs:
            available = list(self.dataset_configs.keys())
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")

        model_config = self.dataset_configs[dataset_name](dataset_params)
        return UNet2DModel(**model_config)

    def create_model_from_config(self, config_path: str) -> UNet2DModel:
        """Load a YAML config and create the corresponding UNet2DModel."""
        cfg       = self.load_config(config_path)
        ds_params = cfg.get('dataset_params', {})
        return self.create_unet_from_dataset(ds_params)

    def get_supported_datasets(self) -> list:
        """Return a list of supported dataset names."""
        return list(self.dataset_configs.keys())

def create_unet(config_path: str) -> UNet2DModel:
    """Create a UNet2DModel directly from a config file."""
    return UNET_MODELS().create_model_from_config(config_path)
