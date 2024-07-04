import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from histogram_equalization import HistogramEqualization
from contrast_enhancement import ContrastEnhancement

def get_data_loaders(train_dir, val_dir, test_dir):

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(227, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        HistogramEqualization(),
        ContrastEnhancement(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(227),
        HistogramEqualization(),
        ContrastEnhancement(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    return train_loader, val_loader, test_loader
