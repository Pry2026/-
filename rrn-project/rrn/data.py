"""
Data loading with augmentation for MNIST, FashionMNIST, and CIFAR-10.

Usage:
    from rrn.data import get_dataloaders
    train_loader, test_loader, input_shape, num_classes = get_dataloaders("cifar10")
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATASETS = ("mnist", "fashionmnist", "cifar10")


def get_dataloaders(
    name: str = "mnist",
    data_dir: str = "./data",
    batch_size: int = 128,
    augment: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, Tuple[int, ...], int]:
    """
    Returns (train_loader, test_loader, input_shape, num_classes).

    input_shape:
      - mnist/fashionmnist: (784,)     (flattened 28×28)
      - cifar10:            (3, 32, 32) (channel-first)
    """
    name = name.lower().replace("-", "").replace("_", "")
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {DATASETS}.")

    if name in ("mnist", "fashionmnist"):
        return _load_grayscale(name, data_dir, batch_size, augment, num_workers)
    else:
        return _load_cifar10(data_dir, batch_size, augment, num_workers)


def _load_grayscale(
    name: str, data_dir: str, batch_size: int, augment: bool, num_workers: int,
) -> Tuple[DataLoader, DataLoader, Tuple[int, ...], int]:
    DatasetCls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST

    mean, std = (0.1307,), (0.3081,)

    train_tf = []
    if augment:
        train_tf += [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
        ]
    train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    test_tf = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train_ds = DatasetCls(data_dir, train=True, download=True,
                          transform=transforms.Compose(train_tf))
    test_ds = DatasetCls(data_dir, train=False, download=True,
                         transform=transforms.Compose(test_tf))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, (784,), 10


def _load_cifar10(
    data_dir: str, batch_size: int, augment: bool, num_workers: int,
) -> Tuple[DataLoader, DataLoader, Tuple[int, ...], int]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = []
    if augment:
        train_tf += [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    train_tf += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    test_tf = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train_ds = datasets.CIFAR10(data_dir, train=True, download=True,
                                transform=transforms.Compose(train_tf))
    test_ds = datasets.CIFAR10(data_dir, train=False, download=True,
                               transform=transforms.Compose(test_tf))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, (3, 32, 32), 10
