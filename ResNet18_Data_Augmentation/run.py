import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from model import get_model
from train import train_model, validate_model
from test import test_model
from inference import plot_metrics, plot_differences
from data_loader import create_data_loaders, k_fold_data_loaders
from utils import load_model_from_checkpoint, perform_inference, save_results, load_config, save_checkpoint, set_seed, \
    load_inference_data


def main():
    config_file = 'config.yaml'
    config = load_config(config_file)

    set_seed(config['training']['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    num_classes = len(config["data"]["classes"])

    train_dataset, val_dataset, test_dataset = create_data_loaders(config, k_fold=True)

    accuracies = {
        'train': [],
        'val': [],
        'test': []
    }
    losses = {
        'train': [],
        'val': [],
        'test': []
    }

    for fold, (train_loader, val_loader) in enumerate(k_fold_data_loaders(train_dataset, config)):
        print(f"Fold {fold + 1}")
        model_instance = get_model(config["model"], num_classes).to(device)
        trained_model, metrics = train_model(model_instance, train_loader, val_loader, config)

        plot_metrics(metrics, fold + 1)

        val_loss, val_accuracy = validate_model(trained_model, val_loader, nn.CrossEntropyLoss(), device)

        test_loader = create_data_loaders(config, k_fold=False)[2]  # Assuming test_loader is the third return value
        test_accuracy, precision, recall, f1, roc_auc, test_loss_fold = test_model(trained_model, test_loader, device)

        accuracies['train'].append(metrics['train_loss'][-1])
        accuracies['val'].append(val_accuracy)
        accuracies['test'].append(test_accuracy)

        losses['train'].append(metrics['train_loss'][-1])
        losses['val'].append(metrics['val_loss'][-1])
        losses['test'].append(test_loss_fold)

    plot_accuracies_losses(accuracies, losses, config['training']['k_folds'])

    plot_differences(trained_model, test_loader, device, config["data"]["classes"])

    model_save_path = config['training']['checkpoint_dir'] + '/updated_best_model_With_AUG.pth'
    torch.save(trained_model.state_dict(), model_save_path)

    print("Training completed successfully!")


def plot_accuracies_losses(accuracies, losses, k_folds):
    folds = range(1, k_folds + 1)

    plt.figure(figsize=(12, 6))

    # Plot training, validation, and test accuracies
    plt.subplot(1, 2, 1)
    plt.plot(folds, accuracies['train'], 'bo-', label='Training Accuracy')
    plt.plot(folds, accuracies['val'], 'ro-', label='Validation Accuracy')
    plt.plot(folds, accuracies['test'], 'go-', label='Test Accuracy')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.title('Accuracies')
    plt.legend()

    # Plot training, validation, and test losses
    plt.subplot(1, 2, 2)
    plt.plot(folds, losses['train'], 'bo-', label='Training Loss')
    plt.plot(folds, losses['val'], 'ro-', label='Validation Loss')
    plt.plot(folds, losses['test'], 'go-', label='Test Loss')
    plt.xlabel('Fold')
    plt.ylabel('Loss')
    plt.title('Losses')
    plt.legend()

    plt.tight_layout()
    plt.show()

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define your model architecture here

    def forward(self, x):
        # Define the forward pass
        return x


checkpoint_path = 'path/to/checkpoint.pth'
data_path = 'path/to/inference_data.pth'
output_path = 'path/to/results.pth'
config_file = 'path/to/config.yaml'
checkpoint_dir = 'path/to/checkpoints'
output_filename = 'model_checkpoint.pth'

# Load configuration
config = load_config(config_file)
print("Configuration loaded:", config)

# Set seed for reproducibility
set_seed(config['seed'])

# Load model from checkpoint
model = load_model_from_checkpoint(checkpoint_path, MyModel)

# Load inference data
data = load_inference_data(data_path)

# Create data loaders
batch_size = config.get('batch_size', 32)
num_workers = config.get('num_workers', 4)
train_loader = create_data_loaders(data, batch_size, num_workers)

# Perform inference
results = perform_inference(model, data)

# Save results
save_results(results, output_path)

# Save model checkpoint
save_checkpoint(model, checkpoint_dir, output_filename)

if __name__ == '__main__':
    main()
