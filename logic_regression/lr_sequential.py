import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Training data
X = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
Y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0]])

# 2. Define model using nn.Sequential
model = nn.Sequential(
    nn.Linear(1, 1),     # Input size: 1, Output size: 1
    nn.Sigmoid()         # Activation: sigmoid for binary classification
)

# 3. Loss function and optimizer
criterion = nn.BCELoss()                # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training loop
epochs = 500
losses = []

for epoch in range(epochs):
    # Forward pass
    y_pred = model(X)

    # Compute loss
    loss = criterion(y_pred, Y)
    losses.append(loss.item())

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}: loss={loss.item():.6f}')


with torch.no_grad():
    test_input = torch.tensor([[1.5]])
    prediction = model(test_input)
    print(f"Predicted probability: {prediction.item():.4f}")
    print(f"Predicted class: {int(prediction.item() > 0.5)}")