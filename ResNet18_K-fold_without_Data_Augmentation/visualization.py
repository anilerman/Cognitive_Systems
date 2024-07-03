# visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_metrics(metrics):
    epochs = range(1, len(metrics['train_loss']) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics['train_accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, metrics['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_differences(model, dataloader, device, classes):
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
    plt.show()
