from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from train_evaluate import train_and_evaluate
import torch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

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