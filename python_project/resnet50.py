from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from train_evaluate import train_and_evaluate
import torch

#Define hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 10
num_classes = 3  # Briefcase, Axe, Box

def train_and_eval(train_dataset, test_dataset, weights, resnet_name='NoNameResnet50'):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training is being performed on: {device}")
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    optomizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    return train_and_evaluate(
        model, optomizer, criterion, device, train_dataset, test_dataset, num_epochs, batch_size, resnet_name,
    )