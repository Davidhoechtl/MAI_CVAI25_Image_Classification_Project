from pathlib import Path
import downloader
from tensorflow import keras as ks
import tensorflow as tf
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import Adam
import time
import numpy as np
import matplotlib.pyplot as plt
from augmentaion import apply_augmentation_to_set
from activaion_map import preprocess_image, generate_activation_map, show_activation_map
import os

epochs = 10

# Download data if it doesn't exist
if downloader.data_exists() == False:
    downloader.download_images()

# Define paths for data
workspaceFold = Path(__file__).resolve().parent.parent
train_folder = workspaceFold / "data" / "train"
test_folder = workspaceFold / "data" / "test"
augmented_data_output_folder = workspaceFold / "data" / "train_augmented"
# Define the folder containing the images
images_folder_for_activation_map = workspaceFold / "images_for_actmap"

# Apply augmentations to the dataset
apply_augmentation_to_set(train_folder/"Axe", augmented_data_output_folder/"Axe")
apply_augmentation_to_set(train_folder/"Box", augmented_data_output_folder/"Box")
apply_augmentation_to_set(train_folder/"Briefcase", augmented_data_output_folder/"Briefcase")

# Data augmentation and preprocessing
train_datagen = ks.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ks.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Load datasets
train_generator = train_datagen.flow_from_directory(
    train_folder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)
test_generator = test_datagen.flow_from_directory(
    test_folder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)
train_generator_augmented = train_datagen.flow_from_directory(
    augmented_data_output_folder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)


# Function to create a ResNet50 model
def create_resnet50(weights, num_classes):
    print(f"Creating ResNet50 model with weights: {weights}")
    base_model = ResNet50(weights=weights, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Measure overall runtime
overall_start_time = time.time()

# Experiment 1 - Plain ResNet50 (Randomly Initialized)
print("Experiment 1: Plain ResNet50 (Randomly Initialized)")
plain_model_start_time = time.time()
plain_model = create_resnet50(weights=None, num_classes=train_generator.num_classes)
plain_model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
plain_history = plain_model.fit(train_generator, epochs=epochs, validation_data=test_generator)
plain_model_end_time = time.time()
print(f"Plain ResNet50 Runtime: {plain_model_end_time - plain_model_start_time:.2f} seconds")

# Experiment 2 - Pre-trained ResNet50 (Transfer Learning)
print("\nExperiment 2: Pre-trained ResNet50 (Transfer Learning)")
pretrained_model_start_time = time.time()
pretrained_model = create_resnet50(weights="imagenet", num_classes=train_generator.num_classes)
pretrained_model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
pretrained_history = pretrained_model.fit(train_generator, epochs=epochs, validation_data=test_generator)
pretrained_model_end_time = time.time()
print(f"Pre-trained ResNet50 Runtime: {pretrained_model_end_time - pretrained_model_start_time:.2f} seconds")

# Experiment 3 - Plain ResNet50 with Augmented Data
print("\nExperiment 3: Plain ResNet50 (Augmented Data)")
plain_aug_model_start_time = time.time()
plain_aug_model = create_resnet50(weights=None, num_classes=train_generator_augmented.num_classes)
plain_aug_model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
plain_aug_history = plain_aug_model.fit(train_generator_augmented, epochs=epochs, validation_data=test_generator)
plain_aug_model_end_time = time.time()
print(f"Plain ResNet50 (Augmented) Runtime: {plain_aug_model_end_time - plain_aug_model_start_time:.2f} seconds")

# Experiment 4 - Pre-trained ResNet50 with Augmented Data
print("\nExperiment 4: Pre-trained ResNet50 (Augmented Data)")
pretrained_aug_model_start_time = time.time()
pretrained_aug_model = create_resnet50(weights="imagenet", num_classes=train_generator_augmented.num_classes)
pretrained_aug_model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
pretrained_aug_history = pretrained_aug_model.fit(train_generator_augmented, epochs=epochs, validation_data=test_generator)
pretrained_aug_model_end_time = time.time()
print(f"Pre-trained ResNet50 (Augmented) Runtime: {pretrained_aug_model_end_time - pretrained_aug_model_start_time:.2f} seconds")

# Print overall runtime
overall_end_time = time.time()
print(f"\nOverall Runtime: {overall_end_time - overall_start_time:.2f} seconds")

# Comparisons
print("\nComparison of Plain vs Pre-trained ResNet50:")
for epoch in range(10):
    print(f"Epoch {epoch+1}: Plain Loss = {plain_history.history['loss'][epoch]:.4f}, "
          f"Pre-trained Loss = {pretrained_history.history['loss'][epoch]:.4f}, "
          f"Plain Accuracy = {plain_history.history['accuracy'][epoch]:.2f}, "
          f"Pre-trained Accuracy = {pretrained_history.history['accuracy'][epoch]:.2f}")

print("\nComparison of Plain vs Pre-trained ResNet50 (Augmented Data):")
for epoch in range(10):
    print(f"Epoch {epoch+1}: Plain Augmented Loss = {plain_aug_history.history['loss'][epoch]:.4f}, "
          f"Pre-trained Augmented Loss = {pretrained_aug_history.history['loss'][epoch]:.4f}, "
          f"Plain Augmented Accuracy = {plain_aug_history.history['accuracy'][epoch]:.2f}, "
          f"Pre-trained Augmented Accuracy = {pretrained_aug_history.history['accuracy'][epoch]:.2f}")

print("\nComparison of Augmented vs Non-Augmented Results:")
for epoch in range(10):
    print(f"Epoch {epoch+1}: Non-Augmented Loss = {plain_history.history['loss'][epoch]:.4f}, "
          f"Augmented Loss = {plain_aug_history.history['loss'][epoch]:.4f}, "
          f"Non-Augmented Accuracy = {plain_history.history['accuracy'][epoch]:.2f}, "
          f"Augmented Accuracy = {plain_aug_history.history['accuracy'][epoch]:.2f}")
    


# Classify images and generate activation maps for each model
# models = [
#     ("Plain ResNet50", plain_model),
#     ("Pre-trained ResNet50", pretrained_model),
#     ("Plain ResNet50 (Augmented)", plain_aug_model),
#     ("Pre-trained ResNet50 (Augmented)", pretrained_aug_model),
# ]

# class_names = list(train_generator.class_indices.keys())  # Get class names

# show_activation_map(models, images_folder_for_activation_map, class_names)