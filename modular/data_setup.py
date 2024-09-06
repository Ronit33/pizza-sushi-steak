import os
from typing import Tuple, List

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
) -> Tuple[DataLoader, DataLoader, List[str]]:
    
    # Use ImageFolder to create datasets
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into dataloaders
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        train_data, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader, class_names