import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import KFold

def create_data_loaders(data_config, k_fold=None, fold_idx=None):
    data_dir = data_config['data_dir']
    batch_size = data_config['batch_size']

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.RandomRotation(20),
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

    # Load complete datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test']}

    # If performing K-Fold cross-validation
    if k_fold is not None and fold_idx is not None:
        kfold = KFold(n_splits=k_fold, shuffle=True, random_state=data_config['seed'])
        train_indices, val_indices = list(kfold.split(image_datasets['train']))[fold_idx]

        train_dataset = Subset(image_datasets['train'], train_indices)
        val_dataset = Subset(image_datasets['train'], val_indices)

        # Weighted Random Sampler
        class_counts = [len(train_dataset.targets) - sum(train_dataset.targets),
                        sum(train_dataset.targets)]
        class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = class_weights[train_dataset.targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:


        train_loader = DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
