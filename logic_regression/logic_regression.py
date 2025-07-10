import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Training data
X = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
Y = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])  # Labels: 0 or 1

# 2. Initialize parameters
w = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)

# 3. Sigmoid activation
def sigmoid(z):
    return 1 / (1 + torch.exp(-z))

# 4. Logistic regression forward pass
def forward(x):
    return sigmoid(w * x + b)

# 5. Binary Cross Entropy Loss
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-7  # to avoid log(0)
    return -(y_true * torch.log(y_pred + eps) + (1 - y_true) * torch.log(1 - y_pred + eps)).mean()

# 6. Training loop
learning_rate = 0.1
epochs = 500
losses = []

for epoch in range(epochs):
    # Forward
    y_pred = forward(X)

    # Loss
    loss = binary_cross_entropy(Y, y_pred)
    losses.append(loss.item())

    # Backward
    loss.backward()

    # Update
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}: w={w.item():.3f}, b={b.item():.3f}, loss={loss.item():.6f}')

# 7. Plot loss
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Logistic Regression Loss over Epochs')
plt.grid()
plt.show()
