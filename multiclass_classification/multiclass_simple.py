import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Multiclass training data: inputs and corresponding class labels (0, 1, or 2)
X = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
Y = torch.tensor([0, 0, 1, 2, 2])  # integer labels (not one-hot)

model = nn.Sequential(
    nn.Linear(1, 3)  # Output 3 logits (one per class)
    # Don't use softmax manually — CrossEntropyLoss applies it internally.
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs = 500
losses = []

for epoch in range(epochs):
    y_pred = model(X)                 # shape: [5, 3]
    loss = criterion(y_pred, Y)       # Y is shape [5], not one-hot
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

with torch.no_grad():
    test_input = torch.tensor([[1.5]])
    logits = model(test_input)            # shape [1, 3]
    predicted_class = torch.argmax(logits, dim=1).item()

    print(f"Raw logits: {logits}")
    print(f"Predicted class: {predicted_class}")
