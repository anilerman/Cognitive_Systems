# train.py

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from utils import save_checkpoint
import model as model_module
from data_loader import create_data_loaders
from utils import load_config

def train_model(model, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['gamma'])

    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    metrics = {
        'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []
        # Add additional metrics as needed
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct / total

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_accuracy'].append(train_accuracy)
        metrics['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, config['training']['checkpoint_dir'], 'best_model.pth')

    return model, metrics

def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

if __name__ == '__main__':
    config_file = 'config.yaml'
    config = load_config(config_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance = model_module.get_model(config["model"], len(config["data"]["classes"])).to(device)
    train_loader, val_loader, _ = create_data_loaders(config['data'])
    train_model(model_instance, train_loader, val_loader, config)
