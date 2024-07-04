import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import KFold
import yaml
import random
import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np


# Function to set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Function to load configuration
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

class ImbalancedDatasetSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset[idx][1]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

def create_data_loaders(dataset, batch_size, num_workers):
    train_loader = DataLoader(
        dataset,
        sampler=ImbalancedDatasetSampler(dataset),
        batch_size=batch_size,
        num_workers=num_workers
    )
    return train_loader

def k_fold_data_loaders(dataset, k, batch_size, num_workers):
    # Implement k-fold data loaders if necessary
    pass

# Function to create data loaders
def create_data_loaders(config, k_fold=False):
    mean = config['data']['mean']
    std = config['data']['std']

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = ImageFolder(root=config['data']['train_data_dir'], transform=train_transform)
    val_dataset = ImageFolder(root=config['data']['val_data_dir'], transform=val_transform)
    test_dataset = ImageFolder(root=config['data']['test_data_dir'], transform=val_transform)

    if not k_fold:
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                                  sampler=ImbalancedDatasetSampler(train_dataset), num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=8, pin_memory=True)

        return train_loader, val_loader, test_loader

    return train_dataset, val_dataset, test_dataset

# Function to create k-fold data loaders
def k_fold_data_loaders(train_dataset, config):
    kf = KFold(n_splits=config['training']['k_folds'], shuffle=True, random_state=config['training']['seed'])
    for fold, (train_index, val_index) in enumerate(kf.split(np.arange(len(train_dataset)))):
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)

        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], sampler=train_sampler, num_workers=2)
        val_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], sampler=val_sampler, num_workers=2)

        yield train_loader, val_loader
