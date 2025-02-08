"""CIFAR-10 data loading with 5-class subset and augmentation transforms."""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 5-class subset: airplane, automobile, bird, cat, deer
CLASSES     = ['airplane', 'automobile', 'bird', 'cat', 'deer']
CLASS_IDS   = [0, 1, 2, 3, 4]          # CIFAR-10 class indices
NUM_CLASSES = len(CLASSES)

# ImageNet-style normalisation values computed on CIFAR-10
_MEAN = (0.4914, 0.4822, 0.4465)
_STD  = (0.2470, 0.2435, 0.2616)

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def _filter(dataset: datasets.CIFAR10) -> Subset:
    """Return a Subset containing only the 5 target classes."""
    indices = [i for i, (_, label) in enumerate(dataset) if label in CLASS_IDS]
    return Subset(dataset, indices)


def get_dataloaders(data_dir: str = './data', batch_size: int = 64,
                    num_workers: int = 2):
    """Download CIFAR-10 (if needed) and return (train_loader, val_loader)."""
    train_full = datasets.CIFAR10(data_dir, train=True,  download=True, transform=TRAIN_TRANSFORMS)
    test_full  = datasets.CIFAR10(data_dir, train=False, download=True, transform=TEST_TRANSFORMS)

    train_sub = _filter(train_full)
    test_sub  = _filter(test_full)

    train_loader = DataLoader(train_sub, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(test_sub,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f'Train samples : {len(train_sub):,}')
    print(f'Val   samples : {len(test_sub):,}')
    return train_loader, val_loader
