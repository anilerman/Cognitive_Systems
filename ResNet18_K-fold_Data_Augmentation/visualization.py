import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
import os

def plot_metrics(metrics, save_path=None, save_format='png'):
    epochs = range(1, len(metrics['train_loss']) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, metrics['val_loss'], 'ro-', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_loss.{save_format}", format=save_format)
        print(f"Loss plot saved at {save_path}_loss.{save_format}")
    else:
        plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metrics['train_accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, metrics['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(f"{save_path}_accuracy.{save_format}", format=save_format)
        print(f"Accuracy plot saved at {save_path}_accuracy.{save_format}")
    else:
        plt.show()

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
            ax.imshow(img, cmap='gray')
            ax.set_title(f"True: {classes[misclassified_labels[i]]}\nPred: {classes[misclassified_predictions[i]]}")
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f"{save_path}_misclassified.{save_format}", format=save_format, bbox_inches='tight')
        print(f"Misclassified images plot saved at {save_path}_misclassified.{save_format}")
    else:
        plt.show()

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

def plot_roc_curve(model, dataloader, device, classes, save_path='results/roc_curve.png'):
    model.eval()
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    n_classes = len(classes)
    one_hot_labels = np.eye(n_classes)[all_labels]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(one_hot_labels[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {classes[i]}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved at {save_path}")
    plt.show()

def plot_confusion_matrix(model, dataloader, device, classes, save_path='results/confusion_matrix.png'):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved at {save_path}")
    plt.show()
