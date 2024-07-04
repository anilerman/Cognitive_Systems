import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from data_loader import create_data_loaders, load_config, set_seed


def get_test_loader(config):
    test_dataset = create_data_loaders(config, k_fold=False)[2]  # Assuming test_dataset is the third return value

    batch_size = config['training']['batch_size']
    num_workers = 8 if torch.cuda.is_available() else 2  # Adjust based on your system

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return test_loader

def test_model(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    roc_auc = roc_auc_score(all_labels, all_predictions, average='weighted', multi_class='ovr')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'ROC AUC: {roc_auc:.4f}')

    return accuracy, precision, recall, f1, roc_auc, test_loss
def main():
    config_file = 'config.yaml'
    config = load_config(config_file)

    set_seed(config['training']['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device

    num_classes = len(config["data"]["classes"])

    test_loader = get_test_loader(config)


if __name__ == '__main__':
    main()
