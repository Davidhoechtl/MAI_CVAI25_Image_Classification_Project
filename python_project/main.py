# cool code will be written here
from pathlib import Path
import downloader
from augmentaion import apply_augmentation_to_set

# download data
if downloader.data_exists() == False:
    downloader.download_images()

# create a train set that contains augmented data
workspaceFold = Path(__file__).resolve().parent
data_folder = workspaceFold / "data" / "train"
output_folder = workspaceFold / "data" / "train_augmented"
apply_augmentation_to_set(data_folder/"axe", output_folder/"axe")
apply_augmentation_to_set(data_folder/"box", output_folder/"box")
apply_augmentation_to_set(data_folder/"briefcase", output_folder/"briefcase")
