import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import os

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # Load and resize the image
    img_array = img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img, img_array

def generate_activation_map(model, img_array, original_img, class_names):
    # Get the last convolutional layer
    last_conv_layer = model.get_layer("conv5_block3_out")
    
    # Create a model that outputs both the feature maps and the predictions
    activation_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
    
    # Get the feature maps and predictions
    feature_maps, predictions = activation_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class]
    confidence = float(predictions[0][predicted_class])
    
    print(f"Prediction: {predicted_class_name} with {confidence:.2%} confidence")
    print(f"Feature map shape: {feature_maps.shape}")
    
    # Get the weights of the final dense layer for the predicted class
    class_weights = model.layers[-1].get_weights()[0][:, predicted_class]
    print(f"Class weights shape: {class_weights.shape}")
    
    # Check if we need to limit the feature maps based on available weights
    max_features = min(feature_maps.shape[3], len(class_weights))
    print(f"Using {max_features} features for activation map")
    
    # Create activation map
    activation_map = np.zeros(feature_maps[0].shape[:2])
    for i in range(max_features):
        activation_map += class_weights[i] * feature_maps[0, :, :, i]
    
    # Apply nonlinearity to enhance strong activations (gamma correction)
    gamma = 0.5  # Lower gamma makes bright spots more prominent
    activation_map = np.power(activation_map, gamma)
    
    # Normalize the activation map
    activation_map = np.maximum(activation_map, 0)
    if np.max(activation_map) > 0:
        activation_map /= np.max(activation_map)
    
    print(f"Activation map min: {np.min(activation_map)}, max: {np.max(activation_map)}")
    
    # Resize the activation map to match the original image size
    activation_map = np.resize(activation_map, (original_img.size[1], original_img.size[0]))
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(original_img)
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Overlay the activation map on the original image
    ax2.imshow(original_img)
    heatmap = ax2.imshow(activation_map, cmap='hot', alpha=0.7)  # Using 'hot' colormap with higher alpha
    ax2.set_title(f"Predicted: {predicted_class_name} ({confidence:.1%})")
    ax2.axis('off')
    
    # Add a color bar
    cbar = fig.colorbar(heatmap, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Intensity')
    
    plt.tight_layout()
    plt.show()
    
def show_activation_map(models, image_folder_path, class_names):
    for model_name, model in models:
        print(f"\nGenerating activation maps for {model_name}...")
        for image_name in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, image_name)
            original_img, img_array = preprocess_image(image_path)

            print(f"Classifying image: {image_name}")
            generate_activation_map(model, img_array, original_img, class_names)