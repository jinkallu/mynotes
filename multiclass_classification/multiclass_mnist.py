import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# 2. Define a model (Flatten → Linear → 10 classes)
model = nn.Sequential(
    nn.Flatten(),         # 28x28 → 784
    nn.Linear(784, 10)    # 10 logits (for digits 0–9)
)

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()                      # applies softmax internally
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. Training loop
epochs = 5
losses = []

for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)                        # logits: [batch, 10]
        loss = criterion(outputs, labels)              # labels: [batch]
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}")

# 5. Plot training loss trend
plt.plot(losses)
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.title("Training Loss on MNIST")
plt.grid(True)
plt.show()

# 6. Evaluate on test set
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"\n🧠 Test Accuracy: {accuracy:.2f}%")
