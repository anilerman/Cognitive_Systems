import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import get_data_loaders
from model import AlexNet
import torch.optim.lr_scheduler as lr_scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def train_model(model, train_loader, val_loader, num_epochs, device):
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#     # learning rate scheduler
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#     best_val_accuracy = 0.0
#     for epoch in range(num_epochs):
#
#         model.train()
#         for batch_idx, (data, targets) in enumerate(train_loader):
#             data, targets = data.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs = model(data)
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
#
#             if batch_idx % 20 == 0:
#                 print(f'Epoch [{epoch}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
#         scheduler.step()  # Update learning rate
#         val_accuracy = calculate_accuracy(model, val_loader, device)
#         print(f'Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.4f}')
#
#         if val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             torch.save(model.state_dict(), 'best_model_last.pth')
def train_model(model, train_loader, val_loader, num_epochs, device):
    loss_val = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_val_accuracy = 0.0

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
                loss_val.append(loss.item())
                val_accuracy = calculate_accuracy(model, val_loader, device)
                val_accuracies.append(val_accuracy)

        train_losses.append(running_loss / len(train_loader))
        scheduler.step()  # Update learning rate



        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model_last.pth')




def calculate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total
