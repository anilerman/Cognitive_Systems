# test.py

import torch
from data_loader import create_data_loaders
import model as model_module
from utils import load_config, check_gpu
import test

def main():
    config_file = 'config.yaml'
    config = load_config(config_file)

    check_gpu()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(config["data"]["classes"])

    model_instance = model_module.get_model(config["model"], num_classes).to(device)
    _, _, test_loader = create_data_loaders(config['data'])
    test.test_model(model_instance, test_loader, device)

def test_model(model, data_loader, device):
    model.eval()
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    accuracy = running_corrects.double() / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

if __name__ == '__main__':
    main()
