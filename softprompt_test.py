import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

# Define the MNIST classifier
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# Define the soft prompt generator
class SoftPromptGenerator(nn.Module):
    def __init__(self):
        super(SoftPromptGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

# Data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Device configuration
device = torch.device("mps")

# Stage 1: Train MNIST Classifier
mnist_classifier = MNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mnist_classifier.parameters(), lr=0.001)

def train_classifier(model, train_loader, criterion, optimizer, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

print("Training MNIST Classifier...")
train_classifier(mnist_classifier, train_loader, criterion, optimizer)

# save the model
torch.save(mnist_classifier.state_dict(), 'mnist_classifier.pth')

# Evaluation function for classifier only
def evaluate_classifier(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Evaluate classifier only
classifier_accuracy = evaluate_classifier(mnist_classifier, test_loader)
print(f"Accuracy of classifier only: {classifier_accuracy:.2f}%")

# Stage 2: Train Soft Prompt Generator
soft_prompt_generator = SoftPromptGenerator().to(device)
optimizer = optim.Adam(soft_prompt_generator.parameters(), lr=0.001)

def train_soft_prompt(generator, classifier, train_loader, criterion, optimizer, num_epochs=1):
    generator.train()
    classifier.eval()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            soft_prompt = generator(data)
            enhanced_input = data + soft_prompt
            output = classifier(enhanced_input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

print("Training Soft Prompt Generator...")
train_soft_prompt(soft_prompt_generator, mnist_classifier, train_loader, criterion, optimizer)
# Evaluation function for classifier with soft prompt
def evaluate_with_soft_prompt(classifier, generator, test_loader):
    classifier.eval()
    generator.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            soft_prompt = generator(data)
            enhanced_input = data + soft_prompt
            output = classifier(enhanced_input)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Evaluate classifier with soft prompt
soft_prompt_accuracy = evaluate_with_soft_prompt(mnist_classifier, soft_prompt_generator, test_loader)
print(f"Accuracy of classifier with soft prompt: {soft_prompt_accuracy:.2f}%")


# Visualization function
def visualize_soft_prompt(classifier, generator, image, index):
    classifier.eval()
    generator.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)
        soft_prompt = generator(image)
        enhanced_image = image + soft_prompt

    # Move tensors to CPU, detach, and convert to numpy arrays
    image = image.cpu().detach().squeeze().numpy()
    soft_prompt = soft_prompt.cpu().detach().squeeze().numpy()
    enhanced_image = enhanced_image.cpu().detach().squeeze().numpy()

    # Normalize the arrays to [0, 255] range
    def normalize(arr):
        arr_min, arr_max = arr.min(), arr.max()
        return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)

    image = normalize(image)
    soft_prompt = normalize(soft_prompt)
    enhanced_image = normalize(enhanced_image)

    # Create a new image with three subimages side by side
    width = image.shape[1]
    height = image.shape[0]
    combined_image = Image.new('L', (width * 3, height))

    # Paste the three images
    combined_image.paste(Image.fromarray(image), (0, 0))
    combined_image.paste(Image.fromarray(soft_prompt), (width, 0))
    combined_image.paste(Image.fromarray(enhanced_image), (width * 2, 0))

    # Save the combined image
    combined_image.save(f'soft_prompt_visualization_{index}.png')

# Visualize results
print("Generating visualizations...")
for i, (data, target) in enumerate(test_loader):
    if i < 5:  # Show first 5 examples
        visualize_soft_prompt(mnist_classifier, soft_prompt_generator, data[i], i)
    else:
        break

print("Visualization complete. Check the current directory for saved images.")