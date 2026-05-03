import os
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import torch


def get_dataset(dataset_config: dict, train: bool = True):
    name = dataset_config.get('name', '').lower()
    if name == 'mnist':
        img_size = dataset_config.get('im_size', 32)
        data_root = dataset_config.get('im_path', './data') or './data'
        os.makedirs(data_root, exist_ok=True)
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return datasets.MNIST(
            root=data_root,
            train=train,
            transform=transform,
            download=True
        )
    elif name == 'cifar':
        img_size = dataset_config.get('im_size', 32)
        data_root = dataset_config.get('im_path', './data') or './data'
        os.makedirs(data_root, exist_ok=True)
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return datasets.CIFAR10(
            root=data_root,
            train=train,
            transform=transform,
            download=True
        )
    elif name == 'celeba':
        img_size = dataset_config.get('im_size', 64)
        data_root = dataset_config.get('im_path', './celeba_data') or './celeba_data'
        os.makedirs(data_root, exist_ok=True)
        
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        if train:
            split = 'train'
        else:
            split = 'test'
            
        return datasets.CelebA(
            root=data_root,
            split=split,
            transform=transform,
            download=False
        )
    elif name == 'ffhq':
        img_size = dataset_config.get('im_size', 128)
        data_root = dataset_config.get('im_path', './FFHQ')

        
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # ImageFolder expects structure: data_root/class_name/images.png
        # For FFHQ with flat structure, use parent directory
        return datasets.ImageFolder(
            root=data_root,
            transform=transform
        )
    elif name == 'imagenet':
        img_size = dataset_config.get('im_size', 256)
        data_root = dataset_config.get('im_path', './imagenet')
        
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Determine which split to load
        if train:
            split_dir = os.path.join(data_root, 'train')
        else:
            split_dir = os.path.join(data_root, 'test')
        
        # ImageFolder expects structure: split_dir/class_name/images.jpg
        return datasets.ImageFolder(
            root=split_dir,
            transform=transform
        )
    elif name == 'lsun':
        img_size = dataset_config.get('im_size', 256)
        data_root = dataset_config.get('im_path', './data/lsun')

        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # im_path = 'data/lsun'; ImageFolder treats 'churches/' as the class dir
        return datasets.ImageFolder(root=data_root, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")


def get_train_val_dataloaders(dataset_config: dict,
                              train: bool = True,
                              train_batch_size: int = 64,
                              val_batch_size: int = 64,
                              val_num_samples: int = 5000,
                              shuffle: bool = True,
                              **loader_kwargs):
    """
    Returns training and validation DataLoaders by splitting the dataset.

    Args:
        dataset_config: dict with dataset params (must include 'name', etc.)
        train: whether to use training split of dataset for splitting
        train_batch_size: batch size for training loader
        val_batch_size: batch size for validation loader
        val_num_samples: number of samples in validation set
        shuffle: whether to shuffle training loader
        loader_kwargs: additional DataLoader args
    """
    name = dataset_config.get('name', '').lower()
    
    if name == 'celeba':
        train_ds = get_dataset(dataset_config, train=True)
        val_config = dataset_config.copy()
        val_ds_full = _get_celeba_validation(val_config)
        
        if val_num_samples < len(val_ds_full):
            seed = dataset_config.get('seed', 0)
            remaining_val = len(val_ds_full) - val_num_samples
            val_ds, _ = random_split(
                val_ds_full,
                [val_num_samples, remaining_val],
                generator=torch.Generator().manual_seed(seed)
            )
        else:
            val_ds = val_ds_full
        
        print(f"CelebA Train set size: {len(train_ds):,}")
        print(f"CelebA Validation subset size: {len(val_ds):,} (from {len(val_ds_full):,} total)")
        print(f"Total CelebA images used: {len(train_ds) + len(val_ds):,}")
        
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=shuffle,
            **loader_kwargs
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            **loader_kwargs
        )
        return train_loader, val_loader
    
    elif name == 'imagenet':
        train_ds = get_dataset(dataset_config, train=True)
        val_config = dataset_config.copy()
        val_ds_full = _get_imagenet_validation(val_config)
        
        if val_num_samples < len(val_ds_full):
            seed = dataset_config.get('seed', 0)
            remaining_val = len(val_ds_full) - val_num_samples
            val_ds, _ = random_split(
                val_ds_full,
                [val_num_samples, remaining_val],
                generator=torch.Generator().manual_seed(seed)
            )
        else:
            val_ds = val_ds_full
        
        print(f"ImageNet Train set size: {len(train_ds):,}")
        print(f"ImageNet Validation subset size: {len(val_ds):,} (from {len(val_ds_full):,} total)")
        print(f"Total ImageNet images used: {len(train_ds) + len(val_ds):,}")
        
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=shuffle,
            **loader_kwargs
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            **loader_kwargs
        )
        return train_loader, val_loader
    
    elif name == 'lsun':
        ds_full = get_dataset(dataset_config, train)

        seed = dataset_config.get('seed', 0)
        total = len(ds_full)
        if val_num_samples >= total:
            raise ValueError(f"val_num_samples ({val_num_samples}) >= dataset size ({total})")

        train_count = total - val_num_samples
        indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_num_samples]

        train_ds = Subset(ds_full, train_indices)
        val_ds = Subset(ds_full, val_indices)

        print(f"LSUN Train subset size: {len(train_ds):,}")
        print(f"LSUN Validation subset size: {len(val_ds):,}")
        print(f"Total LSUN images used: {len(train_ds) + len(val_ds):,}")

        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=shuffle,
            **loader_kwargs
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            **loader_kwargs
        )
        return train_loader, val_loader

    elif name == 'ffhq':
        # Load full FFHQ dataset
        ds_full = get_dataset(dataset_config, train)
        
        # Take subset of 1000 images for training
    
        seed = dataset_config.get('seed', 0)
        

        total = len(ds_full)
        if val_num_samples >= total:
            raise ValueError(f"val_num_samples ({val_num_samples}) >= dataset size ({total})")

        
        # Create indices for train and val
        train_count = total - val_num_samples
        indices = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).tolist()
        train_indices = indices[:train_count]
        val_indices = indices[train_count:train_count + val_num_samples]
        
        train_ds = Subset(ds_full, train_indices)
        val_ds = Subset(ds_full, val_indices)
        
        print(f"FFHQ Train subset size: {len(train_ds):,}")
        print(f"FFHQ Validation subset size: {len(val_ds):,}")
        print(f"Total FFHQ images used: {len(train_ds) + len(val_ds):,}")
        
        
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=shuffle,
            **loader_kwargs
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            **loader_kwargs
        )
        return train_loader, val_loader
        
    else:
        # Original logic for MNIST and CIFAR-10
        ds_full = get_dataset(dataset_config, train)
        total = len(ds_full)
        if val_num_samples >= total:
            raise ValueError(f"val_num_samples ({val_num_samples}) >= dataset size ({total})")
        train_count = total - val_num_samples
        seed = dataset_config.get('seed', 0)
        train_ds, val_ds = random_split(
            ds_full,
            [train_count, val_num_samples],
            generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            shuffle=shuffle,
            **loader_kwargs
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            **loader_kwargs
        )
        return train_loader, val_loader


def _get_celeba_validation(dataset_config):
    """Get CelebA validation split"""
    img_size = dataset_config.get('im_size', 64)
    data_root = dataset_config.get('im_path', './celeba_data') or './celeba_data'
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return datasets.CelebA(
        root=data_root,
        split='valid',
        transform=transform,
        download=False
    )


def _get_imagenet_validation(dataset_config):
    """Get ImageNet validation split"""
    img_size = dataset_config.get('im_size', 256)
    data_root = dataset_config.get('im_path', './imagenet')
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_dir = os.path.join(data_root, 'validation')
    
    return datasets.ImageFolder(
        root=val_dir,
        transform=transform
    )
