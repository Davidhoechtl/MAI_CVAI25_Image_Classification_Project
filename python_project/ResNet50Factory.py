from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, LeakyReLU, Input

def create_resnet50(weights, num_classes, name):
    print(f"Creating ResNet50 model with weights: {weights}")
    base_model = ResNet50(weights=weights, include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions, name=name)
    return model

def create_resnet50_kernel1x1(weights, num_classes):
    base = ResNet50(weights=weights, include_top=False, input_shape=(224, 224, 3))
    x = base.get_layer('conv3_block4_out').output

    x = Conv2D(1024, (1, 1), strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=predictions, name="custom_resnet50_kernel1x1")
    return model

def create_resnet50_kernel3x3(weights, num_classes):
    base = ResNet50(weights=weights, include_top=False, input_shape=(224, 224, 3))
    x = base.get_layer('conv3_block4_out').output

    x = Conv2D(512, (3, 3), strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=predictions, name="custom_resnet50_kernel3x3")
    return model

def create_resnet50_kernel3x3_v2(weights, num_classes):
    base = ResNet50(weights=weights, include_top=False, input_shape=(224, 224, 3))
    x = base.get_layer('conv3_block4_out').output

    x = Conv2D(1024, (3, 3), strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=predictions, name="custom_resnet50_kernel3x3_v2")
    return model

def create_resnet50_frozen_layers(weights='imagenet', num_classes=3):
    base = ResNet50(weights=weights, include_top=False, input_shape=(224, 224, 3))
    x = base.get_layer('conv3_block4_out').output
    """
    Freeze all layers up to and including conv2_block3_out.
    """
    for layer in base.layers:
        if 'conv1' in layer.name or 'conv2_' in layer.name:
            layer.trainable = False

    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=predictions, name="custom_resnet50_frozen")
    return model
