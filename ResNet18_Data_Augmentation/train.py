import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

def train_model(model, train_loader, val_loader, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scaler = torch.cuda.amp.GradScaler()
    device = config['device']

    metrics = defaultdict(list)
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(config['training']['num_epochs']):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{config['training']['num_epochs']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Check if current validation loss is the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    # Load the best model state
    model.load_state_dict(best_model)

    return model, metrics


def validate_model(model, val_loader, criterion, device):
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
