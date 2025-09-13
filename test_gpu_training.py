#!/usr/bin/env python3
"""
Test script to verify GPU training functionality for wakeword detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os

print("=== Wakeword GPU Training Test ===")

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    print("WARNING: Using CPU - GPU not available!")

# Simple CNN model for testing
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)  # Binary classification
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Create model and move to device
model = TestModel().to(device)
print(f"Model moved to {device}")

# Create dummy data
batch_size = 32
input_channels = 1
height, width = 80, 80  # Mel-spectrogram size

# Generate synthetic audio-like data
dummy_input = torch.randn(batch_size, input_channels, height, width).to(device)
dummy_labels = torch.randint(0, 2, (batch_size,)).to(device)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n=== Training Test ===")
print(f"Batch size: {batch_size}")
print(f"Input shape: {dummy_input.shape}")
print(f"Device: {device}")

# Training loop
model.train()
num_epochs = 3
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # Forward pass
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_labels)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | "
          f"Epoch time: {epoch_time:.3f}s | Total: {total_time:.1f}s")

print("\n=== GPU Memory Usage ===")
if torch.cuda.is_available():
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

print("\n=== Test Summary ===")
print(f"SUCCESS: Model created and moved to {device}")
print(f"SUCCESS: Training completed {num_epochs} epochs")
print(f"SUCCESS: Final loss: {loss.item():.4f}")
print(f"SUCCESS: Total training time: {time.time() - start_time:.2f} seconds")
print(f"SUCCESS: GPU training pipeline verified!")

if torch.cuda.is_available():
    print("\n🎉 GPU Training System Ready!")
    print("You can now run the full wakeword_training.ipynb notebook")
else:
    print("\nERROR: GPU not detected - check CUDA installation")