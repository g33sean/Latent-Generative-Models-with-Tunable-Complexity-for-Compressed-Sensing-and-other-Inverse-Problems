import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch

def get_dataset(dataset_config: dict, train: bool = True):
    name = dataset_config.get('name', '').lower()
    
    if name == 'celeba':
        img_size = dataset_config.get('im_size', 64)
        data_root = dataset_config.get('im_path', './celeba_data') or './celeba_data'
        os.makedirs(data_root, exist_ok=True)
        
        # Build as list, then conditionally add normalize for train
        tfms = [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ]
        
        if train:
            tfms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))  # train-only
        
        transform = transforms.Compose(tfms)
        
        # Map train flag to CelebA splits
        split = 'train' if train else 'test'
        
        return datasets.CelebA(
            root=data_root,
            split=split,
            transform=transform,
            download=False  # Assume data is already downloaded
        )
    
    elif name == 'celeba_hq':
        img_size = 64  # Fixed size for CelebA-HQ
        data_root = dataset_config.get('im_path', './celeba_hq_data') or './celeba_hq_data'
        
        # Transform to resize to 64x64 and keep values in [0, 1]
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()  # This converts to [0, 1] range
        ])
        
        # Use ImageFolder to load from directory structure
        # Assumes data is organized as: data_root/train/ and data_root/test/
        # or just data_root/ with all images
        if os.path.exists(os.path.join(data_root, 'train')) and os.path.exists(os.path.join(data_root, 'test')):
            # If train/test split exists
            split_dir = 'train' if train else 'test'
            dataset_path = os.path.join(data_root, split_dir)
        else:
            # If no split, use all data
            dataset_path = data_root
        
        return datasets.ImageFolder(
            root=dataset_path,
            transform=transform
        )
    
    elif name == 'ffhq':
        img_size = 256  # Fixed size for FFHQ
        data_root = dataset_config.get('im_path', './ffhq_data') or './ffhq_data'
        
        # Transform to resize to 64x64 and keep values in [0, 1]
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()  # This converts to [0, 1] range
        ])
        
        # Use ImageFolder to load from directory structure
        # Assumes data is organized as: data_root/train/ and data_root/test/
        # or just data_root/ with all images
        if os.path.exists(os.path.join(data_root, 'train')) and os.path.exists(os.path.join(data_root, 'test')):
            # If train/test split exists
            split_dir = 'train' if train else 'test'
            dataset_path = os.path.join(data_root, split_dir)
        else:
            # If no split, use all data
            dataset_path = data_root
        
        return datasets.ImageFolder(
            root=dataset_path,
            transform=transform
        )
    
    else:
        raise ValueError(f"Unsupported dataset: {name}. Supported datasets: celeba, celeba_hq, ffhq")