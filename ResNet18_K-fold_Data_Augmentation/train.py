import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os

from data_loader import create_data_loaders
from model import get_model
from utils import save_checkpoint
from visualization import plot_metrics


import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import os

from data_loader import create_data_loaders
from model import get_model
from utils import save_checkpoint
from visualization import plot_metrics

def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['gamma'])

    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    metrics = {
        'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f},'
              f' Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, config['training']['checkpoint_dir'], f'kfold_model_wit_Aug{epoch+1}.pth')

    return model, metrics

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy


def plot_differences(model, dataloader, device, classes, save_path=None, save_format='png', max_images=5):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            misclassified = predicted != labels

            misclassified_images.extend(images[misclassified].cpu())
            misclassified_labels.extend(labels[misclassified].cpu())
            misclassified_predictions.extend(predicted[misclassified].cpu())


            if len(misclassified_images) >= max_images:
                break

    num_images = len(misclassified_images)
    if num_images == 0:
        print("No misclassified images found.")
        return

    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 3))

    for i in range(num_rows * num_cols):
        ax = axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i % num_cols]
        if i < num_images:
            img = misclassified_images[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            ax.set_title(f"True: {classes[misclassified_labels[i]]}\nPred: {classes[misclassified_predictions[i]]}")
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path = f"{save_path}.{save_format}"
        fig.savefig(save_path, format=save_format, bbox_inches='tight')
        print(f"Plot saved at {save_path}")

    plt.show()

if __name__ == "__main__":
    import json

    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config['data'])

    # Get model
    model = get_model(config['model'], num_classes=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train model
    trained_model, metrics = train_model(model, train_loader, val_loader, config)

    # Evaluate on test set
    test_loss, test_accuracy = evaluate_model(trained_model, test_loader, torch.nn.CrossEntropyLoss(), device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

    # Plot metrics
    plot_metrics(metrics, save_path='results/metrics', save_format='png')

    # Plot misclassified images
    classes = ['NORMAL', 'PNEUMONIA']
    plot_differences(trained_model, test_loader, device, classes, save_path='results/misclassified', save_format='png', max_images=5)
