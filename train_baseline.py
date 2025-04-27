import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import UCF101Dataset
from src.models.c3d_baseline import C3DBaseline
import matplotlib.pyplot as plt

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
EPOCHS = 10
BATCH_SIZE = 4
LR = 0.001
NUM_WORKERS = 4

if __name__ == "__main__":
    # Dataset and DataLoader
    dataset = UCF101Dataset(root_dir="data/UCF101")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    # Model, loss, optimizer
    model = C3DBaseline().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Track loss and accuracy
    train_losses = []
    train_accuracies = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataset)
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")

    # Plotting results
    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, marker='o')
    plt.title('Training Loss over Epochs (Baseline)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.savefig('loss_plot_baseline.png')
    plt.close()

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_accuracies, marker='o')
    plt.title('Training Accuracy over Epochs (Baseline)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid()
    plt.savefig('accuracy_plot_baseline.png')
    plt.close()
