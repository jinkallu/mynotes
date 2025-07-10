import torch
import matplotlib.pyplot as plt

# Data (convert to tensors)
X = torch.tensor([-2, -1, 1, 2], dtype=torch.float32)
W_original = 1.0
B_Original = 0.0
Y = W_original * X + B_Original  # Y = wx + b

# Initialize parameters with gradient tracking
w = torch.randn((), requires_grad=True)  # scalar weight
b = torch.randn((), requires_grad=True)  # scalar bias

learning_rate = 0.01
epochs = 500
losses = []

# Forward pass function
def forward(x, w, b):
    return w * x + b    

# Loss function (Mean Squared Error)
def loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Training loop     
for epoch in range(epochs):
    # Forward pass
    y_predicted = forward(X, w, b)
    
    # Calculate loss
    l = loss(Y, y_predicted)
    losses.append(l.item())
    
    # Backward pass
    l.backward()  # Compute gradients
    
    # Update parameters
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Zero gradients for the next iteration
    w.grad.zero_()
    b.grad.zero_()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}: w={w.item():.3f}, b={b.item():.3f}, loss={l.item():.8f}')

# Plot loss curve
fig, ax = plt.subplots()

ax.plot(range(1, epochs + 1), losses, label='Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.grid(True)
ax.legend()
fig.savefig("Loss_over_Epochs_pytorch.png")