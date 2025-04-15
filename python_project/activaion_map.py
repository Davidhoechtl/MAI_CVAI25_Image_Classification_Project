import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
import os

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # Load and resize the image
    img_array = img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img, img_array

# Function to generate and visualize activation maps
def generate_activation_map(model, img_array, original_img, class_names):
    # Get the last convolutional layer
    last_conv_layer = model.get_layer("conv5_block3_out")  # Adjust layer name for ResNet50

    # Create a model that outputs the feature maps of the last convolutional layer
    activation_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])

    # Get the feature maps and predictions
    feature_maps, predictions = activation_model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class]

    # Get the weights of the predicted class
    class_weights = model.layers[-1].get_weights()[0][:, predicted_class]

    # Compute the activation map
    activation_map = np.dot(feature_maps[0], class_weights)

    # Normalize the activation map
    activation_map = np.maximum(activation_map, 0)
    activation_map /= np.max(activation_map)

    # Resize the activation map to match the original image size
    activation_map = np.resize(activation_map, (original_img.size[1], original_img.size[0]))

    # Overlay the activation map on the original image
    plt.imshow(original_img)
    plt.imshow(activation_map, cmap="jet", alpha=0.5)  # Overlay with transparency
    plt.title(f"Predicted: {predicted_class_name}")
    plt.axis("off")
    plt.show()
    
def show_activation_map(models, image_path, class_names):
    for model_name, model in models:
        print(f"\nGenerating activation maps for {model_name}...")
        for image_name in os.listdir(image_path):
            image_path = os.path.join(image_path, image_name)
            original_img, img_array = preprocess_image(image_path)

            print(f"Classifying image: {image_name}")
            generate_activation_map(model, img_array, original_img, class_names)