import torch
import yaml
from pathlib import Path
from diffusers import AutoencoderKL
from typing import Dict, Any


class VAE_MODELS:
    """Factory class for creating AutoencoderKL models based on configuration.
    Supports CelebA, MNIST, and CIFAR-10 datasets, with an extensible design for future datasets.
    """
    
    def __init__(self):
        self.dataset_configs = {
            'CELEBA':   self._get_celeba_config,
            'FFHQ':     self._get_ffhq_config,
            'IMAGENET': self._get_img_config,
            'MNIST':    self._get_mnist_config,
            'CIFAR':    self._get_cifar10_config,
            'LSUN':     self._get_lsun_config,
        }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
     
     
    def _get_img_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get ImageNet-specific model configuration."""
        im_channels = dataset_params.get('im_channels', 3)
        im_size     = dataset_params.get('im_size', 256)
        latent_channels    = dataset_params.get('latent_channels', 12)
 
        return {
            'in_channels':        im_channels,
            'out_channels':       im_channels,
            'latent_channels':    latent_channels,
            'sample_size':        im_size,
            'block_out_channels': (192, 384, 640),
            'norm_num_groups':    32,
            'down_block_types':   ("DownEncoderBlock2D",) * 3,
            'up_block_types':     ("UpDecoderBlock2D",)  * 3,
            'layers_per_block':   2,
            'act_fn':             "silu",
            'mid_block_add_attention': True,  
        }
    
    def _get_ffhq_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get FFHQ-specific model configuration."""
        im_channels = dataset_params.get('im_channels', 3)
        im_size     = dataset_params.get('im_size', 256)
        latent_channels    = dataset_params.get('latent_channels', 12)
 
        return {
            'in_channels':        im_channels,
            'out_channels':       im_channels,
            'latent_channels':    latent_channels,
            'sample_size':        im_size,
            'block_out_channels': (192, 384, 640),
            'norm_num_groups':    32,
            'down_block_types':   ("DownEncoderBlock2D",) * 3,
            'up_block_types':     ("UpDecoderBlock2D",)  * 3,
            'layers_per_block':   2,
            'act_fn':             "silu",
        }
    
    def _get_lsun_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get LSUN-specific model configuration."""
        im_channels = dataset_params.get('im_channels', 3)
        im_size     = dataset_params.get('im_size', 256)
        latent_channels    = dataset_params.get('latent_channels', 12)

        return {
            'in_channels':        im_channels,
            'out_channels':       im_channels,
            'latent_channels':    latent_channels,
            'sample_size':        im_size,
            'block_out_channels': (192, 384, 640),
            'norm_num_groups':    32,
            'down_block_types':   ("DownEncoderBlock2D",) * 3,
            'up_block_types':     ("UpDecoderBlock2D",)  * 3,
            'layers_per_block':   2,
            'act_fn':             "silu",
            'mid_block_add_attention': True,
        }

    def _get_celeba_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get CelebA-specific model configuration."""
        im_channels = dataset_params.get('im_channels', 3)
        im_size     = dataset_params.get('im_size', 64)
        latent_channels    = dataset_params.get('latent_channels', 16)
 
        return {
            'in_channels':        im_channels,
            'out_channels':       im_channels,
            'latent_channels':    latent_channels,
            'sample_size':        im_size,
            'block_out_channels': (128, 256, 512),
            'norm_num_groups':    32,
            'down_block_types':   ("DownEncoderBlock2D",) * 3,
            'up_block_types':     ("UpDecoderBlock2D",)  * 3,
            'layers_per_block':   1,
            'act_fn':             "silu",
        }
        

    def _get_mnist_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get MNIST-specific model configuration (~1.2M params for 1×32×32 images)."""
        im_channels = dataset_params.get('im_channels', 1)
        im_size     = dataset_params.get('im_size',    32)
        latent_channels    = dataset_params.get('latent_channels', 4)
        return {
            'in_channels':        im_channels,
            'out_channels':       im_channels,
            'latent_channels':    latent_channels,
            'sample_size':        im_size,
            'block_out_channels': (16, 32, 64),
            'norm_num_groups':    8,
            'down_block_types':   ("DownEncoderBlock2D",) * 3,
            'up_block_types':     ("UpDecoderBlock2D",)   * 3,
            'layers_per_block':   1,
            'act_fn':             "silu",
        }
    
    def _get_cifar10_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get CIFAR-10 specific model configuration."""
        im_channels = dataset_params.get('im_channels', 3)
        im_size     = dataset_params.get('im_size',    32)
        latent_channels    = dataset_params.get('latent_channels', 4)
        
        if latent_channels==8:
            return {
            'in_channels':        im_channels,
            'out_channels':       im_channels,
            'latent_channels':    latent_channels,
            'sample_size':        32,
            'block_out_channels': (128, 384),
            'norm_num_groups':    32,
            'down_block_types':   ("DownEncoderBlock2D",) * 2,
            'up_block_types':     ("UpDecoderBlock2D",)   * 2,
            'layers_per_block':   2,
            'act_fn':             "silu",
            }
        else:
            return {
            'in_channels':        im_channels,
            'out_channels':       im_channels,
            'latent_channels':    latent_channels,
            'sample_size':        32,
            'block_out_channels': (64, 128, 256),
            'norm_num_groups':    16,
            'down_block_types':   ("DownEncoderBlock2D",) * 3,
            'up_block_types':     ("UpDecoderBlock2D",)   * 3,
            'layers_per_block':   2,
            'act_fn':             "silu",
        }

    def create_autoencoder_from_dataset(self, dataset_params: Dict[str, Any]) -> AutoencoderKL:
        """Create an AutoencoderKL model based on dataset name in dataset_params."""
        name = dataset_params.get('name', None)
        if not name:
            raise ValueError("Please specify dataset_params['name'] (e.g., 'CELEBA', 'MNIST', or 'CIFAR').")
        dataset_name = name.upper()
        if dataset_name not in self.dataset_configs:
            available = list(self.dataset_configs.keys())
            raise ValueError(
                f"Dataset '{dataset_name}' not supported. Available: {available}"
            )
        model_config = self.dataset_configs[dataset_name](dataset_params)
        return AutoencoderKL(**model_config)

    def create_model_from_config(self, config_path: str) -> AutoencoderKL:
        """Load a YAML config and create the corresponding AutoencoderKL."""
        cfg       = self.load_config(config_path)
        ds_params = cfg.get('dataset_params', {})
        return self.create_autoencoder_from_dataset(ds_params)

    def get_supported_datasets(self) -> list:
        """Return a list of supported dataset names."""
        return list(self.dataset_configs.keys())


def create_model(config_path: str) -> AutoencoderKL:
    """Create an AutoencoderKL directly from a config file."""
    return VAE_MODELS().create_model_from_config(config_path)
