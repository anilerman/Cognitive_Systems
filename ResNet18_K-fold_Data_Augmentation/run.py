import os
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

import test
from visualization import plot_metrics, plot_differences, plot_roc_curve, plot_confusion_matrix
import model as model_module
import train
from utils import load_config, set_seed, check_gpu


def create_data_loaders(data_config, k_fold=None, fold_idx=None):
    data_dir = data_config['data_dir']
    batch_size = data_config['batch_size']

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(data_config['image_size']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((data_config['image_size'], data_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((data_config['image_size'], data_config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
    ])

    if k_fold:
        dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        train_idx, val_idx = k_fold[fold_idx]
        train_loader = DataLoader(dataset, batch_size=batch_size,
                                  sampler=torch.utils.data.SubsetRandomSampler(train_idx))
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx))
        test_loader = None
    else:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)

    return epoch_loss, epoch_acc

def plot_correctly_classified_images(model, dataloader, device, classes, save_path=None, save_format='png', max_images=5):
    model.eval()
    correct_images = []
    correct_labels = []
    correct_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = predicted == labels

            correct_images.extend(images[correct].cpu())
            correct_labels.extend(labels[correct].cpu())
            correct_predictions.extend(predicted[correct].cpu())

            if len(correct_images) >= max_images:
                break

    num_images = len(correct_images)
    if num_images == 0:
        print("No correctly classified images found.")
        return

    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))

    for i in range(num_rows * num_cols):
        ax = axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i % num_cols]
        if i < num_images:
            img = correct_images[i].permute(1, 2, 0).numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"True: {classes[correct_labels[i]]}\nPred: {classes[correct_predictions[i]]}")
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_correctly_classified.{save_format}", format=save_format, bbox_inches='tight')
        print(f"Correctly classified images plot saved at {save_path}_correctly_classified.{save_format}")
    else:
        plt.show()

def main():
    config_file = 'config.yaml'
    config = load_config(config_file)

    set_seed(config['training']['seed'])
    check_gpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(config["data"]["classes"])

    train_loader, val_loader, test_loader = create_data_loaders(config['data'])

    # Initialize model
    model = model_module.get_model(config['model'], num_classes).to(device)

    # Train model
    trained_model, metrics = train.train_model(model, train_loader, val_loader, config)

    # Plot metrics
    plot_metrics(metrics, save_path='results/metrics')

    # Evaluate model on validation set
    model.eval()  # Set model to evaluation mode
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_accuracy = evaluate_model(trained_model, val_loader, criterion, device)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')


    (test_accuracy, test_precision, test_recall, test_f1,
     test_roc_auc) = test.test_model(trained_model, test_loader,device)


    plot_correctly_classified_images(trained_model, val_loader, device, config['data']['classes'],
                                     save_path='results/correct_images')


    plot_differences(trained_model, val_loader, device, classes=config['data']['classes'],
                     save_path='results/misclassified_images')

    plot_differences(trained_model, test_loader, device, classes=config['data']['classes'],
                     save_path='results/misclassified')


    plot_roc_curve(trained_model, test_loader, device, classes=config['data']['classes'],
                   save_path='results/roc_curve.png')

    plot_confusion_matrix(trained_model, test_loader, device, classes=config['data']['classes'],
                          save_path='results/confusion_matrix.png')


    k_fold = config['training'].get('k_fold', None)
    if k_fold:
        kfold = KFold(n_splits=k_fold, shuffle=True, random_state=config['training']['seed'])
        dataset = datasets.ImageFolder(os.path.join(config['data']['data_dir'], 'train'))
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"Starting Fold {fold + 1}/{k_fold}")
            train_loader, val_loader, _ = create_data_loaders(config['data'], k_fold=kfold, fold_idx=fold)
            model_instance = model_module.get_model(config["model"], num_classes).to(device)
            trained_model, metrics = train.train_model(model_instance, train_loader, val_loader, config)


            plot_metrics(metrics, save_path=f'results/metrics_fold_{fold + 1}')


if __name__ == '__main__':
    main()
