import os
import cv2
import random
import numpy as np
from PIL import Image, ImageOps

def load_image(path):
    return Image.open(path)

def save_image(image, path):
    image.save(path)

def affine_transform(image, scale=1.0, angle=0, tx=0, ty=0):
    """
    Function to apply a 2D transformation to the image
    :param image: the image that should be transformed
    :param scale: scale_factor
    :param angle: rotates the image by the angle
    :param tx: translates the image on the x (missing spaces will be filled with black)
    :param ty: translates the image on the y (missing spaces will be filled with black)
    :return: transformed image
    """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rows, cols = image_cv.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(image_cv, M, (cols, rows))
    return Image.fromarray(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))

def flip_image_horizontal(image):
    """ Flips the image on the Y axis"""
    return ImageOps.mirror(image)

def flip_image_vertical(image):
    """ Flips the image on the X axis"""
    return ImageOps.flip(image)

def rotate_image(image, angle):
    """ Rotates the image by the angle specified"""
    return affine_transform(image, 1.0, angle, 0,0)

def scale_image(image, scale_factor):
    """scales the image by the scale_factor specified"""
    return affine_transform(image, scale_factor, 0,0,0)

def translate_image(image, translate_x, translate_y):
    """translates the image by x,y"""
    return affine_transform(image, 0,0, translate_x, translate_y)

def apply_augmentation_to_set(image_folder, output_folder):
    """
    10% horizontal flips
    10% vertical flips
    10% rotations
    10% scales
    10% translations
    50 % no augmentaion
    :param image_folder: input folder
    :param output_folder: output folder
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        image = load_image(image_path)

        rand = random.random()

        # Apply horizontal flip (10%)
        if rand < 0.1:
            aug_image = flip_image_horizontal(image)
            aug_name = f"{os.path.splitext(image_name)[0]}_hflip{os.path.splitext(image_name)[1]}"

        # Apply vertical flip (10%)
        elif rand < 0.2:
            aug_image = flip_image_vertical(image)
            aug_name = f"{os.path.splitext(image_name)[0]}_vflip{os.path.splitext(image_name)[1]}"

        # Apply rotation (20%)
        elif rand < 0.3:
            angle = random.randint(-45, 45)
            aug_image = rotate_image(image, angle)
            aug_name = f"{os.path.splitext(image_name)[0]}_rot{angle}{os.path.splitext(image_name)[1]}"

        # Apply scaling (10%)
        elif rand < 0.4:
            scale_factor = round(random.uniform(0.7, 1.3), 2)
            aug_image = scale_image(image, scale_factor)
            aug_name = f"{os.path.splitext(image_name)[0]}_scale{scale_factor}{os.path.splitext(image_name)[1]}"

        # Apply translation (10%)
        elif rand <= 0.5:
            tx = random.randint(-30, 30)
            ty = random.randint(-30, 30)
            aug_image = translate_image(image, tx, ty)
            aug_name = f"{os.path.splitext(image_name)[0]}_trans{tx}_{ty}{os.path.splitext(image_name)[1]}"

        # no augmentation! (50%)
        else:
            aug_image = image
            aug_name = f"{os.path.splitext(image_name)[0]}_noaug_{os.path.splitext(image_name)[1]}"

        save_path = os.path.join(output_folder, aug_name)
        save_image(aug_image, save_path)