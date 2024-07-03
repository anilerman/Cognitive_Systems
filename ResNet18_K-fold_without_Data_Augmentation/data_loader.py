# data_loader.py

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from utils import load_config

def create_data_loaders(data_config, k_fold=None, fold_idx=None):
    data_dir = data_config['data_dir']
    batch_size = data_config['batch_size']

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
        ]),
    }

    dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])

    if k_fold is not None and fold_idx is not None:
        kfold = KFold(n_splits=k_fold, shuffle=True, random_state=data_config['seed'])
        train_indices, val_indices = list(kfold.split(dataset))[fold_idx]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
