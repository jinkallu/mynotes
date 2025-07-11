import torch
import matplotlib.pyplot as plt

# 1. Training data: input x and 3 class labels (0, 1, 2)
X = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])  # shape: [5, 1]
Y = torch.tensor([0, 0, 1, 2, 2])                        # integer labels

# 2. Initialize weights and bias manually (for 1 input → 3 classes)
w = torch.randn((1, 3), requires_grad=True)  # shape: [in_features, num_classes]
b = torch.randn((1, 3), requires_grad=True)  # shape: [1, num_classes]

# 3. Define forward pass (manual softmax regression)
def forward(x):
    logits = x @ w + b  # shape: [batch, 3]
    return logits  # softmax will be applied in loss

# 4. CrossEntropyLoss (manual)
def cross_entropy_loss(logits, labels):
    # logits: [batch, num_classes], labels: [batch]
    # Apply softmax
    max_logits = torch.max(logits, dim=1, keepdim=True).values  # for stability
    exp_logits = torch.exp(logits - max_logits)
    softmax_probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)

    # Pick log prob of correct class
    batch_indices = torch.arange(len(labels))
    correct_probs = softmax_probs[batch_indices, labels]

    loss = -torch.log(correct_probs + 1e-8).mean()
    return loss

# 5. Training loop
learning_rate = 0.1
epochs = 1000
losses = []

for epoch in range(epochs):
    logits = forward(X)             # raw scores
    loss = cross_entropy_loss(logits, Y)
    losses.append(loss.item())

    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# 6. Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Multiclass Softmax Regression Loss")
plt.grid(True)
plt.show()