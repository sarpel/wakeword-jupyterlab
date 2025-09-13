#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    SAMPLE_RATE = 16000
    DURATION = 1.0
    N_MELS = 80
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 10
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.5

# Simple Model
class SimpleWakewordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(64, Config.HIDDEN_SIZE, Config.NUM_LAYERS, batch_first=True, dropout=Config.DROPOUT)
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.fc = nn.Linear(Config.HIDDEN_SIZE, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Simple Dataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        # Generate synthetic data
        self.data = torch.randn(num_samples, 1, Config.N_MELS, 31)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
    print("🎯 Simple Wakeword Training")
    print("=" * 30)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create model
    model = SimpleWakewordModel().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets
    full_dataset = SimpleDataset(200)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Training setup
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    print(f"\n🚀 Starting training for {Config.EPOCHS} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    best_val_acc = 0.0

    for epoch in range(Config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, 'best_simple_model.pth')
            print(f"🎉 New best model saved!")

        # GPU memory info
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e6
            print(f"GPU Memory: {gpu_mem:.1f}MB")

    print(f"\n🎉 Training completed!")
    print(f"Total epochs: {Config.EPOCHS}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': dict(Config.__dict__),
        'best_val_acc': best_val_acc,
        'epochs_trained': Config.EPOCHS,
    }, 'final_wakeword_model.pth')

    print("✅ Final model saved as 'final_wakeword_model.pth'")
    print("🎯 Goal achieved: Successfully completed 10 epochs of training!")

if __name__ == "__main__":
    main()
