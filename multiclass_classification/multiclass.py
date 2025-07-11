import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Define transform: convert image to tensor & normalize pixel values to [0,1]
transform = transforms.ToTensor()

# Load the MNIST training dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Create a DataLoader to sample batches
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,  # show 8 images
    shuffle=True
)

# Get one batch
images, labels = next(iter(train_loader))  # images shape: [8, 1, 28, 28]

# Plot the batch
plt.figure(figsize=(10, 3))
for i in range(8):
    plt.subplot(1, 8, i+1)
    plt.imshow(images[i][0], cmap='gray')  # images[i][0] = 28x28
    plt.title(f"Label: {labels[i].item()}")
    plt.axis('off')

plt.suptitle("Sample MNIST Digits and Their Labels")
plt.savefig("sample_data.png")
plt.show()