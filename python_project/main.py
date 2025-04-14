from pathlib import Path
import downloader
from train_evaluate import train_and_evaluate
from torchvision import datasets, transforms, models
from resnet50 import train, num_epochs
from augmentaion import apply_augmentation_to_set


# Download data if it doesn't exist
if downloader.data_exists() == False:
    downloader.download_images()

# Define paths for data
workspaceFold = Path(__file__).resolve().parent
train_folder = workspaceFold / "data" / "train"
test_folder = workspaceFold / "data" / "test"

augmented_data_output_folder = workspaceFold / "data" / "train_augmented"
apply_augmentation_to_set(train_folder/"axe", augmented_data_output_folder/"axe")
apply_augmentation_to_set(train_folder/"box", augmented_data_output_folder/"box")
apply_augmentation_to_set(train_folder/"briefcase", augmented_data_output_folder/"briefcase")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 for ResNet50
    transforms.ToTensor(),         # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])


# Load datasets
train_dataset = datasets.ImageFolder(train_folder, transform=transform)
test_dataset = datasets.ImageFolder(test_folder, transform=transform)


# Experiment 1 - Plain ResNet50 (Randomly Initialized)
print("Experiment 1: Plain ResNet50 (Randomly Initialized)")
plain_train_losses, plain_test_accuracies = train(train_dataset, test_dataset, weights=None, resnet_name="Plain ResNet50")

# Experiment 2 - Pre-trained ResNet50 (Transfer Learning)
print("\nExperiment 2: Pre-trained ResNet50 (Transfer Learning)")
pretrained_train_losses, pretrained_test_accuracies = train(train_dataset, test_dataset, weights=models.ResNet50_Weights.IMAGENET1K_V1, resnet_name="Pre-trained ResNet50")

# Compare Results
print("\nComparison of Plain vs Pre-trained ResNet50:")
print("Epoch\tPlain Loss\tPre-trained Loss\tPlain Accuracy\tPre-trained Accuracy")
for epoch in range(num_epochs):
    print(f"{epoch+1}\t{plain_train_losses[epoch]:.4f}\t\t{pretrained_train_losses[epoch]:.4f}\t\t{plain_test_accuracies[epoch]:.2f}%\t\t{pretrained_test_accuracies[epoch]:.2f}%")