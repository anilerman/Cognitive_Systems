import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

def plot_metrics(metrics, fold=None):
    epochs = range(1, len(metrics['train_loss']) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss {f"for Fold {fold}" if fold else ""}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to plot bar graphs of accuracies and losses
def plot_accuracies_losses(accuracies, losses, k_folds):
    x = np.arange(k_folds)
    width = 0.2

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    ax[0].bar(x - width, accuracies['train'], width, label='Train Accuracy')
    ax[0].bar(x, accuracies['val'], width, label='Val Accuracy')
    ax[0].bar(x + width, accuracies['test'], width, label='Test Accuracy')
    ax[0].set_xlabel('Fold')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Train, Validation, and Test Accuracies per Fold')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f'Fold {i+1}' for i in range(k_folds)])
    ax[0].legend()

    ax[1].bar(x - width, losses['train'], width, label='Train Loss')
    ax[1].bar(x, losses['val'], width, label='Val Loss')
    ax[1].bar(x + width, losses['test'], width, label='Test Loss')
    ax[1].set_xlabel('Fold')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Train, Validation, and Test Losses per Fold')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f'Fold {i+1}' for i in range(k_folds)])
    ax[1].legend()

    plt.show()

# Function to plot the differences (misclassified images)
def plot_differences(model, test_loader, device, classes):
    model.eval()
    differences = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            misclassified = (predicted != labels).cpu().numpy()
            if any(misclassified):
                differences.extend(zip(images.cpu().numpy(), labels.cpu().numpy(), predicted.cpu().numpy(), misclassified))

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    for i, (image, true_label, pred_label, misclassified) in enumerate(differences[:10]):
        ax = axes[i]
        image = np.transpose(image, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        ax.set_title(f"True: {classes[true_label]}, Pred: {classes[pred_label]}")
        ax.axis('off')

    plt.suptitle('Misclassified Images')
    plt.show()
