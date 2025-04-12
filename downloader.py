import os
import sys
import random
import shutil

import pandas as pd


# Ensure the script is run from the correct directory
repo_path = os.path.abspath(os.getenv("PYTHONPATH", ""))
if repo_path and repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from pathlib import Path
from openimages.download import download_images
from math import ceil

workspaceFold = Path(__file__).resolve().parent

# Check if the workspace folder exists
if not workspaceFold.exists():
    raise FileNotFoundError(f"Workspace folder does not exist: {workspaceFold}")

# Check if the data folder exists
data_dir = workspaceFold / "data"
if data_dir.exists():
    print(f"Data folder already exists: {data_dir}")
    print("Downloading images...")
else:
    print("Data folder does not exist. Creating a new one.")
    print("Downloading images...")
    data_dir.mkdir(parents=True, exist_ok=True)

# Downloading images from Open Images dataset
download_images(
    os.path.join(workspaceFold, "data"),
    ["Axe", "Briefcase", "Box",], 
    os.path.join(workspaceFold, "exclusions.txt")
)


train_ration = 0.75
val_ration = test_ration = 0.125

image_exts = [".jpg", ".jpeg", ".png"]

# Creating target folders
for split in ["train", "val", "test"]:
    (data_dir / split).mkdir(parents=True, exist_ok=True)

for class_folder in data_dir.iterdir():
    images_path = class_folder / "images"
    if not images_path.exists():
        continue
    
    class_name = class_folder.name
    all_images = [img for img in images_path.iterdir() if img.suffix.lower() in image_exts]
    random.shuffle(all_images)

    total = len(all_images)
    train_end = ceil(total * train_ration)
    val_end = train_end + ceil(total * val_ration)

    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }

    for split, files in splits.items():
        dest_dir = data_dir / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy(file, dest_dir / file.name)
    
    # Remove the original images and class folder after copying
    shutil.rmtree(images_path)
    shutil.rmtree(class_folder)


print("✅ Dataset split completed.")