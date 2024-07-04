import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from data_loader import get_data_loaders
from model import AlexNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_metrics(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_images = []
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_images.extend(data.cpu())
    accuracy = correct / total
    return accuracy, all_predictions, all_targets, all_images
