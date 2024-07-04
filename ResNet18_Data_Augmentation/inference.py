import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_config, load_model_from_checkpoint, load_inference_data
from utils import perform_inference
from utils import save_results

def inference_model(model, data):
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


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

def plot_heatmap(model, layer_name='layer1.0.conv1'):
    activation = None
    for name, layer in model.named_modules():
        if name == layer_name:
            activation = layer
            break

    if activation is None:
        print(f"No layer named {layer_name} found in the model")
        return

    dummy_input = torch.randn(1, 3, 224, 224)
    activation_output = None

    def hook_fn(m, i, o):
        nonlocal activation_output
        activation_output = o

    hook = activation.register_forward_hook(hook_fn)
    model(dummy_input)
    hook.remove()

    if activation_output is None:
        print("Failed to get activation output")
        return

    heatmap_data = activation_output[0, 0].cpu().detach().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title('Activation Heatmap')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()


def main():
    # Load config file
    config_file = 'config.yaml'
    config = load_config(config_file)

    # Example usage of perform_inference function
    model = load_model_from_checkpoint(config['model_checkpoint_path'])
    inference_data = load_inference_data(config['inference_data_path'])
    predictions = perform_inference(model, inference_data, config)

    # Process predictions or save results as needed
    save_results(predictions, config['results_path'])

    print("Inference completed successfully!")


def plot_differences(model, test_loader, device, classes, config):  # Pass config as an argument
    model.eval()
    differences = []

    mean = config['data']['mean']
    std = config['data']['std']

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
        image = np.clip(image * std + mean, 0, 1)  # replace 'mean' and 'std' with actual values
        ax.imshow(image)
        ax.set_title(f"True: {classes[true_label]}, Pred: {classes[pred_label]}")
        ax.axis('off')

    plt.suptitle('Misclassified Images')
    plt.show()

if __name__ == "__main__":
    main()
