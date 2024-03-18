import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader

num_classes = 10
num_epochs = 10
batch_size = 32

# Transforms
transform = models.EfficientNet_B0_Weights.DEFAULT.transforms()

train_data = datasets.CIFAR10(
    root=".",
    train=True,
    download=True,
    transform=transform,
)

test_data = datasets.CIFAR10(
    root=".",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Load the pre-trained EfficientNet-B0 model
model = models.efficientnet_b0(weights="DEFAULT")

# Freeze all the parameters in the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with a new one
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, num_classes)

# Define the number of classes
num_classes = 10

# Print the model architecture
print(model)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train(num_epochs, model, device, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
        
        # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
    
    # Print the average loss for each epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
# Evaluate your model
def eval(model, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
        
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train(num_epochs, model, device, criterion, optimizer)
    eval(model, device)