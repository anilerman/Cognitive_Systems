import torch
import matplotlib.pyplot as plt
import numpy as np
from data_loader import get_data_loaders
from inference import calculate_metrics
from model import AlexNet
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_model(train_dir, val_dir, test_dir, precision_score, recall_score, f1_score, roc_auc_score):
    model = AlexNet().to(device)
    _, _, test_loader = get_data_loaders(train_dir, val_dir, test_dir)

    model.load_state_dict(torch.load('best_model_last.pth'))
    test_accuracy, test_predictions, test_targets, test_images = calculate_metrics(model, test_loader, device)

    print(f'Accuracy on test set: {test_accuracy:.4f}')
    print(f'Precision on test set: {precision_score(test_targets, test_predictions):.4f}')
    print(f'Recall on test set: {recall_score(test_targets, test_predictions):.4f}')
    print(f'F1-score on test set: {f1_score(test_targets, test_predictions):.4f}')
    print(f'AUC-ROC on test set: {roc_auc_score(test_targets, test_predictions):.4f}')
    return test_accuracy, test_predictions, test_targets, test_images
   # num_images_to_plot = 5
   #  fig, axes = plt.subplots(1, num_images_to_plot, figsize=(15, 3))
   #
   #  for i in range(num_images_to_plot):
   #      image = test_images[i].cpu().numpy().transpose((1, 2, 0))
   #      mean = np.array([0.485, 0.456, 0.406])
   #      std = np.array([0.229, 0.224, 0.225])
   #      image = std * image + mean
   #      image = np.clip(image, 0, 1)
   #
   #      axes[i].imshow(image)
   #      axes[i].set_title(f'Actual: {test_targets[i]}, Predicted: {test_predictions[i]}')
   #      axes[i].axis('off')
   #
   #  plt.tight_layout()
   #  plt.show()
