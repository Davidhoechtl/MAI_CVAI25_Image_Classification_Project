from pathlib import Path
import downloader
from torchvision import datasets, transforms, models
from resnet50 import train_and_eval, num_epochs
from augmentaion import apply_augmentation_to_set
import time


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
train_dataset_augmented = datasets.ImageFolder(augmented_data_output_folder, transform=transform)


overall_start_time = time.time()

# Experiment 1 - Plain ResNet50 (Randomly Initialized)
plain_model_start_time = time.time()
plain_train_losses, plain_test_accuracies = train_and_eval(train_dataset, test_dataset, weights=None, resnet_name="Plain ResNet50")
plain_model_end_time = time.time()
print(f"Plain ResNet50 Runtime: {plain_model_end_time - plain_model_start_time:.2f} seconds")

# Experiment 2 - Pre-trained ResNet50 (Transfer Learning)
pretrained_model_start_time = time.time()
pretrained_train_losses, pretrained_test_accuracies = train_and_eval(train_dataset, test_dataset, weights=models.ResNet50_Weights.IMAGENET1K_V1, resnet_name="Pre-trained ResNet50")
pretrained_model_end_time = time.time()
print(f"Pre-trained ResNet50 Runtime: {pretrained_model_end_time - pretrained_model_start_time:.2f} seconds")


# Experiment 3 - Plain ResNet50 with Augmented Data
plain_aug_model_start_time = time.time()
plain_train_losses_aug, plain_test_accuracies_aug = train_and_eval(
    train_dataset_augmented, test_dataset, weights=None, resnet_name="Plain ResNet50 (Augmented)"
)
plain_aug_model_end_time = time.time()
print(f"Plain ResNet50 (Augmented) Runtime: {plain_aug_model_end_time - plain_aug_model_start_time:.2f} seconds")

# Experiment 4 - Pre-trained ResNet50 with Augmented Data
pretrained_aug_model_start_time = time.time()
pretrained_train_losses_aug, pretrained_test_accuracies_aug = train_and_eval(
    train_dataset_augmented, test_dataset, weights=models.ResNet50_Weights.IMAGENET1K_V1, resnet_name="Pre-trained ResNet50 (Augmented)"
)
pretrained_aug_model_end_time = time.time()
print(f"Pre-trained ResNet50 (Augmented) Runtime: {pretrained_aug_model_end_time - pretrained_aug_model_start_time:.2f} seconds")


# Compare Results for non-augmented data
print("\nComparison of Plain vs Pre-trained ResNet50:")
print("Epoch\tPlain Loss\tPre-trained Loss\tPlain Accuracy\tPre-trained Accuracy")
for epoch in range(num_epochs):
    print(f"{epoch+1}\t{plain_train_losses[epoch]:.4f}\t\t{pretrained_train_losses[epoch]:.4f}\t\t{plain_test_accuracies[epoch]:.2f}%\t\t{pretrained_test_accuracies[epoch]:.2f}%")
    

# Compare Results for Augmented Data
print("\nComparison of Plain vs Pre-trained ResNet50 (Augmented Data):")
print("Epoch\tPlain Loss\tPre-trained Loss\tPlain Accuracy\tPre-trained Accuracy")
for epoch in range(num_epochs):
    print(f"{epoch+1}\t{plain_train_losses_aug[epoch]:.4f}\t\t{pretrained_train_losses_aug[epoch]:.4f}\t\t{plain_test_accuracies_aug[epoch]:.2f}%\t\t{pretrained_test_accuracies_aug[epoch]:.2f}%")
    
    
# Compare Augmented vs Non-Augmented Results
print("\nComparison of Augmented vs Non-Augmented Results:")
print("Epoch\tNon-Augmented Loss\tAugmented Loss\tNon-Augmented Accuracy\tAugmented Accuracy")
for epoch in range(num_epochs):
    print(f"{epoch+1}\t{plain_train_losses[epoch]:.4f}\t\t{plain_train_losses_aug[epoch]:.4f}\t\t{plain_test_accuracies[epoch]:.2f}%\t\t{plain_test_accuracies_aug[epoch]:.2f}%")
    

overall_end_time = time.time()
print(f"Overall Runtime: {overall_end_time - overall_start_time:.2f} seconds")