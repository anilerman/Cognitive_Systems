import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from train import train_model, calculate_accuracy
from inference import calculate_metrics
from model import AlexNet
from data_loader import get_data_loaders
import matplotlib.pyplot as plt
import numpy as np
from test import test_model
from graph import  plot_predictions, plot_confusion_matrix, plot_training_history
# datasets
train_dir = r'C:\Users\40721\Desktop\cognitive_systems\code\chest_xray\train'
val_dir = r'C:\Users\40721\Desktop\cognitive_systems\code\chest_xray\val'
test_dir = r'C:\Users\40721\Desktop\cognitive_systems\code\chest_xray\test'

def main():
    #if it is using gpu or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # using the model
    model = AlexNet().to(device)


    train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir)


    num_epochs = 10
    train_model(model, train_loader, val_loader, num_epochs, device)

    # load the model
    model.load_state_dict(torch.load('best_model_last2.pth'))

    # Evaluate the model on test set and calculate metrics
    test_accuracy, test_predictions, test_targets, test_images = calculate_metrics(model, test_loader, device)

    # Print metrics
    test_model(train_dir, val_dir, test_dir, precision_score, recall_score, f1_score, roc_auc_score)

    # Plot five sample images
    num_images_to_plot = 5
    fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 3))

    plot_predictions(test_images, test_targets, test_predictions)
    plot_confusion_matrix(test_targets, test_predictions)

    train_loader, val_loader, test_loader = get_data_loaders(train_dir, val_dir, test_dir)


    #plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    for i in range(num_images_to_plot):
        image = test_images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        axes[i].imshow(image)
        axes[i].set_title(f'Actual: {test_targets[i]}, Predicted: {test_predictions[i]}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
