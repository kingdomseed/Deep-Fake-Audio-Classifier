from numpy.typing import test
from torch import triplet_margin_loss
from torch.utils.data import DataLoader

import dataset as ds


def create_dataloaders(train_features_path, train_labels_path,
                       dev_features_path, dev_labels_path,
                       test_features_path, batch_size=32, num_workers=2):
    """
    Create DataLoaders for train, dev, and test sets.

    Args:
        train_features_path: Path to train features.pkl
        train_labels_path: Path to train labels.pkl
        dev_features_path: Path to dev features.pkl
        dev_labels_path: Path to dev labels.pkl
        test_features_path: Path to test features.pkl
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader

    Returns:
        train_loader, dev_loader, test_loader
    """
    # TODO 1: Create datasets
    # train_dataset = AudioDeepfakeDataset(...)
    # dev_dataset = AudioDeepfakeDataset(...)
    train_dataset = ds.AudioDeepfakeDataset(train_features_path,
                                            train_labels_path)
    dev_dataset = ds.AudioDeepfakeDataset(dev_features_path, dev_labels_path)
    # TODO 2: Create test dataset (for test, we only have features, no labels)
    # For test, we need a different Dataset class OR modify AudioDeepfakeDataset
    # Hint: You can pass labels_path=None and handle it in __init__
    test_dataset = ds.AudioDeepfakeDataset(test_features_path, None)

    # TODO 3: Create DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=..., shuffle=...,
    #                           num_workers=..., pin_memory=...)
    # dev_loader = DataLoader(...)
    # test_loader = DataLoader(...)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, 
                            shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, dev_loader, test_loader


def make_loader(features_path, labels_path, batch_size=32,
                num_workers=2, shuffle=False):
    dataset = ds.AudioDeepfakeDataset(features_path, labels_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)
