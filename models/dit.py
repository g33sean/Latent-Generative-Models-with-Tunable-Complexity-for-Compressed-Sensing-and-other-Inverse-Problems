import yaml
from pathlib import Path
from diffusers import Transformer2DModel
from typing import Dict, Any

class DIT_MODELS:
    """Factory class for creating Transformer2DModel (DiT) models based on configuration.
    Supports CelebA, MNIST, CIFAR-10, FFHQ, and ImageNet datasets.
    """
    def __init__(self):
        self.dataset_configs = {
            'CELEBA':   self._get_celeba_config,
            'MNIST':    self._get_mnist_config,
            'CIFAR':    self._get_cifar10_config,
            'FFHQ':     self._get_ffhq_config,
            'IMAGENET': self._get_imagenet_config,
        }

    def load_config(self, config_path: str) -> Dict[str, Any]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_ffhq_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        FFHQ DiT config.
        """
        latent_channels = dataset_params.get('latent_channels', 24)
        latent_size     = dataset_params.get('latent_size', 64)   # spatial size of latent
        return {
            'num_attention_heads':        16,
            'attention_head_dim':         64,    # 16 * 64 = 1024 hidden dim
            'in_channels':                latent_channels,
            'out_channels':               latent_channels,
            'num_layers':                 28,    # transformer depth
            'dropout':                    0.0,
            'norm_num_groups':            32,
            'attention_bias':             True,
            'sample_size':                latent_size,
            'patch_size':                 2,     # 64/2 = 32x32 tokens
            'activation_fn':              "gelu-approximate",
            'num_embeds_ada_norm':        1000,  # timestep embedding buckets
            'upcast_attention':           False,
            'norm_type':                  "ada_norm_zero",   # DiT-style adaLN-Zero
            'norm_elementwise_affine':    False,
            'caption_channels':           None,  # unconditional
        }

    def _get_imagenet_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        ImageNet DiT config.
        Latent space: 12 x 32 x 32 (typical SD VAE)
        Patch size 2 → 16x16 = 256 tokens
        ~600M parameters
        """
        latent_channels = dataset_params.get('latent_channels', 12)
        latent_size     = dataset_params.get('latent_size', 32)
        return {
            'num_attention_heads':        16,
            'attention_head_dim':         96,    # 16 * 96 = 1536 hidden dim
            'in_channels':                latent_channels,
            'out_channels':               latent_channels,
            'num_layers':                 28,
            'dropout':                    0.0,
            'norm_num_groups':            32,
            'attention_bias':             True,
            'sample_size':                latent_size,
            'patch_size':                 2,
            'activation_fn':              "gelu-approximate",
            'num_embeds_ada_norm':        1000,
            'upcast_attention':           False,
            'norm_type':                  "ada_norm_zero",
            'norm_elementwise_affine':    False,
            'caption_channels':           None,
        }

    def _get_celeba_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        CelebA DiT config.
        Latent space: 12 x 32 x 32
        Patch size 2 → 16x16 = 256 tokens
        ~300M parameters
        """
        latent_channels = dataset_params.get('latent_channels', 12)
        latent_size     = dataset_params.get('latent_size', 32)
        return {
            'num_attention_heads':        12,
            'attention_head_dim':         64,    # 12 * 64 = 768 hidden dim
            'in_channels':                latent_channels,
            'out_channels':               latent_channels,
            'num_layers':                 24,
            'dropout':                    0.0,
            'norm_num_groups':            32,
            'attention_bias':             True,
            'sample_size':                latent_size,
            'patch_size':                 2,
            'activation_fn':              "gelu-approximate",
            'num_embeds_ada_norm':        1000,
            'upcast_attention':           False,
            'norm_type':                  "ada_norm_zero",
            'norm_elementwise_affine':    False,
            'caption_channels':           None,
        }

    def _get_mnist_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        MNIST DiT config — small model, pixel space (1 x 32 x 32).
        Patch size 2 → 16x16 = 256 tokens
        ~30M parameters
        """
        im_channels = dataset_params.get('im_channels', 1)
        latent_size = dataset_params.get('latent_size', 32)
        return {
            'num_attention_heads':        8,
            'attention_head_dim':         32,    # 8 * 32 = 256 hidden dim
            'in_channels':                im_channels,
            'out_channels':               im_channels,
            'num_layers':                 12,
            'dropout':                    0.0,
            'norm_num_groups':            8,
            'attention_bias':             True,
            'sample_size':                latent_size,
            'patch_size':                 2,
            'activation_fn':              "gelu-approximate",
            'num_embeds_ada_norm':        1000,
            'upcast_attention':           False,
            'norm_type':                  "ada_norm_zero",
            'norm_elementwise_affine':    False,
            'caption_channels':           None,
        }

    def _get_cifar10_config(self, dataset_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        CIFAR DiT config.
        Latent space: 8 x 8 x 8
        Patch size 2 → 4x4 = 16 tokens — use patch_size=1 for more tokens
        ~80M parameters
        """
        latent_channels = dataset_params.get('latent_channels', 8)
        latent_size     = dataset_params.get('latent_size', 8)
        return {
            'num_attention_heads':        8,
            'attention_head_dim':         64,    # 8 * 64 = 512 hidden dim
            'in_channels':                latent_channels,
            'out_channels':               latent_channels,
            'num_layers':                 12,
            'dropout':                    0.0,
            'norm_num_groups':            32,
            'attention_bias':             True,
            'sample_size':                latent_size,
            'patch_size':                 1,     # small latent, keep all tokens
            'activation_fn':              "gelu-approximate",
            'num_embeds_ada_norm':        1000,
            'upcast_attention':           False,
            'norm_type':                  "ada_norm_zero",
            'norm_elementwise_affine':    False,
            'caption_channels':           None,
        }

    def create_dit_from_dataset(self, dataset_params: Dict[str, Any]) -> Transformer2DModel:
        """Create a Transformer2DModel (DiT) based on dataset name in dataset_params."""
        name = dataset_params.get('name')
        if not name:
            raise ValueError("Please specify dataset_params['name'] (e.g., 'FFHQ', 'CELEBA', 'CIFAR').")
        dataset_name = name.upper()
        if dataset_name not in self.dataset_configs:
            available = list(self.dataset_configs.keys())
            raise ValueError(f"Dataset '{dataset_name}' not supported. Available: {available}")

        model_config = self.dataset_configs[dataset_name](dataset_params)
        return Transformer2DModel(**model_config)

    def create_model_from_config(self, config_path: str) -> Transformer2DModel:
        """Load a YAML config and create the corresponding DiT model."""
        cfg       = self.load_config(config_path)
        ds_params = cfg.get('dataset_params', {})
        return self.create_dit_from_dataset(ds_params)

    def get_supported_datasets(self) -> list:
        return list(self.dataset_configs.keys())

def create_dit(config_path: str) -> Transformer2DModel:
    """Create a Transformer2DModel directly from a config file."""
    return DIT_MODELS().create_model_from_config(config_path)
