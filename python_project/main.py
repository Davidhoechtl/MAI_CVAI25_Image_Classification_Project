from pathlib import Path
from tensorflow.python.ops.confusion_matrix import confusion_matrix
import downloader
from tensorflow import keras as ks
import tensorflow as tf
from keras.optimizers import Adam
import time
import numpy as np
import matplotlib.pyplot as plt
from augmentaion import apply_augmentation_to_set
from activaion_map import show_activation_map
import ResNet50Factory
import pandas as pd

print(tf.__version__)
print(tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

epochs = 1

# Define paths for data
workspaceFold = Path(__file__).resolve().parent.parent
train_folder = workspaceFold / "data" / "train"
test_folder = workspaceFold / "data" / "test"
augmented_data_output_folder = workspaceFold / "data" / "train_augmented"
# combined_folder = workspaceFold / "data" / "train_combined"
# cleaned_folder = workspaceFold / "data" / "train_cleaned"
# test_cleaned_folder = workspaceFold / "data" / "test_cleaned"
# Define the folder containing the images
images_folder_for_activation_map = workspaceFold / "python_project/images_for_actmap"

# Download data if it doesn't exist
if downloader.data_exists() == False:
    downloader.download_images()

    # Apply augmentations to the dataset
    apply_augmentation_to_set(train_folder/"Axe", augmented_data_output_folder/"Axe")
    apply_augmentation_to_set(train_folder/"Box", augmented_data_output_folder/"Box")
    apply_augmentation_to_set(train_folder/"Briefcase", augmented_data_output_folder/"Briefcase")

# Data augmentation and preprocessing
image_pipeline = ks.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

# Load datasets
train_generator = image_pipeline.flow_from_directory(
    train_folder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)
test_generator = image_pipeline.flow_from_directory(
    test_folder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)
train_generator_augmented = image_pipeline.flow_from_directory(
    augmented_data_output_folder, target_size=(224, 224), batch_size=32, class_mode="categorical"
)
# train_generator_combined = image_pipeline.flow_from_directory(
#     combined_folder, target_size=(224, 224), batch_size=32, class_mode="categorical"
# )
# train_generator_cleaned= image_pipeline.flow_from_directory(
#     cleaned_folder, target_size=(224, 224), batch_size=32, class_mode="categorical"
# )
# test_cleaned_folder_generator_cleaned= image_pipeline.flow_from_directory(
#     test_cleaned_folder, target_size=(224, 224), batch_size=32, class_mode="categorical"
# )

history_dict  = {}
def evaluate_model(train_gen, test_gen, model):
    print(f"{model.name} start...")
    model_start_time = time.time()
    model.compile(optimizer=Adam(learning_rate=0.002), loss="categorical_crossentropy", metrics=["accuracy"])
    model_history = model.fit(train_gen, epochs=epochs, validation_data=test_gen)
    model_end_time = time.time()
    print(f"{model.name} Runtime: {model_end_time - model_start_time:.2f} seconds")
    history_dict[model.name] = model_history

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# Measure overall runtime
overall_start_time = time.time()

# Experiment 1 - Plain ResNet50 (Randomly Initialized)
plain_model = ResNet50Factory.create_resnet50(weights=None, num_classes=train_generator.num_classes, name="Plain_ResNet50")
evaluate_model(train_generator, test_generator, plain_model)

# evaluate confusion matrix for plain resnet50
y_pred_probs = plain_model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
cm_tensor = confusion_matrix(y_true, y_pred)
cm = cm_tensor.numpy()
class_labels = list(test_generator.class_indices.keys())
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Experiment 2 - Pre-trained ResNet50 (Transfer Learning)
pretrained_model = ResNet50Factory.create_resnet50(weights="imagenet", num_classes=train_generator.num_classes, name="Pre_trained_ResNet50")
evaluate_model(train_generator, test_generator, pretrained_model)
# evaluate confusion matrix for plain resnet50
y_pred_probs = pretrained_model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes
cm_tensor = confusion_matrix(y_true, y_pred)
cm = cm_tensor.numpy()
class_labels = list(test_generator.class_indices.keys())
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=45)
plt.yticks(tick_marks, class_labels)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()
# Experiment 3 - Plain ResNet50 with Augmented Data
plain_aug_model = ResNet50Factory.create_resnet50(weights=None, num_classes=train_generator_augmented.num_classes, name="plain_aug_model")
evaluate_model(train_generator_augmented, test_generator, plain_aug_model)

# Experiment 4 - Pre-trained ResNet50 with Augmented Data
pretrained_aug_model = ResNet50Factory.create_resnet50(weights="imagenet", num_classes=train_generator_augmented.num_classes, name="pretrained_aug_model")
evaluate_model(train_generator, test_generator, pretrained_aug_model)

#Experiment 5 - Custom with additional conv layer (kernel3x3, filters 512, stride 1, leaky relu, padding same)
custom1 = ResNet50Factory.create_resnet50_kernel3x3(weights=None, num_classes=train_generator.num_classes)
evaluate_model(train_generator, test_generator, custom1)

#Experiment 6 - Custom with additional conv layer (kernel 1x1, filters 1024, padding same, stride 1, activation leaky relu)
custom2 = ResNet50Factory.create_resnet50_kernel1x1(weights=None, num_classes=train_generator.num_classes)
evaluate_model(train_generator, test_generator, custom2)

#Experiment 7 - Custom with additional conv layer kernel 3x3, filters 1024, padding same, stride 2, activation leaky relu)
custom3 = ResNet50Factory.create_resnet50_kernel3x3_v2(weights=None, num_classes=train_generator.num_classes)
evaluate_model(train_generator, test_generator, custom3)

#Experiment 8 - Freeze conv2 layers and before
custom4 = ResNet50Factory.create_resnet50_frozen_layers(weights=None, num_classes=train_generator.num_classes)
evaluate_model(train_generator, test_generator, custom4)

# Print overall runtime
overall_end_time = time.time()

print(f"\nOverall Runtime: {overall_end_time - overall_start_time:.2f} seconds")

# Extract val_accuracy for each model
val_acc_dict = {}
for model_name, history in history_dict.items():
    val_acc_dict[model_name] = history.history['val_accuracy']
val_acc_df = pd.DataFrame(val_acc_dict)
val_acc_df.index.name = 'Epoch'
val_acc_df.index += 1  # start epochs from 1 instead of 0
print("\nValidation Accuracy per Epoch:")
print(val_acc_df.round(4))  # round to 4 decimal places for clarity

# Extract training accuracy for each model
train_acc_dict = {}
for model_name, history in history_dict.items():
    train_acc_dict[model_name] = history.history['accuracy']
train_acc_df = pd.DataFrame(train_acc_dict)
train_acc_df.index.name = 'Epoch'
train_acc_df.index += 1  # start epochs from 1
print("\nTraining Accuracy per Epoch:")
print(train_acc_df.round(4))

# Plot validation accuracy per epoch for each model
plt.figure(figsize=(14, 8))
for model_name in val_acc_df.columns:
    plt.plot(val_acc_df.index, val_acc_df[model_name], label=model_name)
plt.title("Validation Accuracy per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Validation Accuracy", fontsize=14)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot training accuracy per epoch for each model
plt.figure(figsize=(14, 8))
for model_name in train_acc_df.columns:
    plt.plot(train_acc_df.index, train_acc_df[model_name], label=model_name)
plt.title("Training Accuracy per Epoch", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Training Accuracy", fontsize=14)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

#Classify images and generate activation maps for each model
models = [
    ("Plain ResNet50", plain_model),
    ("Pre-trained ResNet50", pretrained_model),
    ("Plain ResNet50 (Augmented)", plain_aug_model),
    ("Pre-trained ResNet50 (Augmented)", pretrained_aug_model),
]

class_names = list(train_generator.class_indices.keys())  # Get class names

show_activation_map(models, images_folder_for_activation_map, class_names)