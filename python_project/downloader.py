import sys
import zipfile
import pandas as pd
import os
import requests
import shutil
import glob

from tqdm import tqdm
from pathlib import Path
from io import BytesIO

csv_dir = "../openimages_csv"
output_dir = "../data/train"
os.makedirs(csv_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- Download CSV files ---
download_links = {
    "oidv7-class-descriptions-boxable.csv": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
    "oidv6-train-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
    "validation-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "test-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "train-images-boxable-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
    "validation-images-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
    "test-images-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv"
}

print("🔽 Lade CSV-Dateien herunter...")
for filename, url in download_links.items():
    filepath = Path(csv_dir) / filename
    if not filepath.exists():
        try:
            r = requests.get(url, stream=True)
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ {filename}")
        except Exception as e:
            print(f"❌ Fehler beim Herunterladen von {filename}: {e}")
    else:
        print(f"🔁 {filename} existiert bereits - übersprungen.")

target_classes = ["Axe", "Briefcase", "Box"]
max_images_per_class = 500

# 1. load Label-ID Mapping
classes_df = pd.read_csv(f"{csv_dir}/oidv7-class-descriptions-boxable.csv", header=None, names=["LabelName", "ClassName"])
label_map = dict(zip(classes_df.ClassName, classes_df.LabelName))
target_labels = {cls: label_map[cls] for cls in target_classes}
print("Gefundene Label-IDs:", target_labels)

# 2. load Bounding Box
bbox_train = pd.read_csv(f"{csv_dir}/oidv6-train-annotations-bbox.csv")
bbox_val = pd.read_csv(f"{csv_dir}/validation-annotations-bbox.csv")
bbox_test = pd.read_csv(f"{csv_dir}/test-annotations-bbox.csv")

bbox_df = pd.concat([bbox_train, bbox_val, bbox_test], ignore_index=True)
filtered_bbox = bbox_df[bbox_df['LabelName'].isin(target_labels.values())]

# 3. load pictur urls
meta_train = pd.read_csv(f"{csv_dir}/train-images-boxable-with-rotation.csv", usecols=["ImageID", "OriginalURL"])
meta_val = pd.read_csv(f"{csv_dir}/validation-images-with-rotation.csv", usecols=["ImageID", "OriginalURL"])
meta_test = pd.read_csv(f"{csv_dir}/test-images-with-rotation.csv", usecols=["ImageID", "OriginalURL"])

image_meta = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
image_meta = image_meta.dropna()

# load pictures
for class_name, label_id in target_labels.items():
    class_dir = Path(f"{output_dir}/{class_name}")
    class_dir.mkdir(parents=True, exist_ok=True)

    image_ids = filtered_bbox[filtered_bbox['LabelName'] == label_id]['ImageID'].unique()
    image_ids = image_ids[:max_images_per_class]

    # URLs from IDs
    urls = image_meta[image_meta['ImageID'].isin(image_ids)]

    print(f"🔽 {len(urls)} Bilder werden für '{class_name}' heruntergeladen...")

    for _, row in tqdm(urls.iterrows(), total=len(urls)):
        img_url = row['OriginalURL']
        img_id = row['ImageID']
        file_path = class_dir / f"{img_id}.jpg"
        try:
            r = requests.get(img_url, timeout=5)
            if r.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(r.content)
        except Exception:
            continue # Ignore errors and continue with next image


# === Step 1: Download and Extract ZIP ===
zip_url = "https://storage.googleapis.com/cvstock-932a9-h58gl/1a78s8d2ckm3675v728so2.zip"
axe_dir = Path("../data/train/Axe")

print(f"🔽 Lade zusätzliches Datenpaket von {zip_url} herunter...")
try:
    r = requests.get(zip_url, timeout=10)
    if r.status_code == 200:
        with zipfile.ZipFile(BytesIO(r.content)) as zip_ref:
            zip_ref.extractall(axe_dir)
        print("ZIP erfolgreich extrahiert.")
    else:
        print(f"❌ Fehler beim Herunterladen der ZIP-Datei. Status-Code: {r.status_code}")
except Exception as e:
    print(f"❌ Fehler beim Herunterladen oder Entpacken der ZIP-Datei: {e}")


# === Step 2: Move Images ===
axe_nested_root = axe_dir / "images.cv_1a78s8d2ckm3675v728so2/data"
destination = axe_dir

moved = 0
skipped_existing = 0
skipped_non_images = 0

print("\nVerschiebe Bilder...")
for subfolder in ["train", "test", "val"]:
    subdir = axe_nested_root / subfolder
    if not subdir.exists():
        print(f"Ordner '{subdir}' existiert nicht - wird übersprungen.")
        continue

    for img_path in subdir.rglob("*.*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            target_path = destination / img_path.name
            if not target_path.exists():
                try:
                    shutil.move(str(img_path), str(target_path))
                    moved += 1
                except Exception as e:
                    print(f"❌ Fehler beim Verschieben {img_path.name}: {e}")
            else:
                skipped_existing += 1
        else:
            skipped_non_images += 1

# === Step 3: Optional - Clean Up ===
extracted_zip_folder = axe_dir / "images.cv_1a78s8d2ckm3675v728so2"
try:
    shutil.rmtree(extracted_zip_folder)
    print(f"\n🧹 Ordner '{extracted_zip_folder}' erfolgreich gelöscht.")
except Exception as e:
    print(f"Fehler beim Löschen von '{extracted_zip_folder}': {e}")

print(f"\n✅ Zusammenfassung:")
print(f"  📦  {moved} Bilder verschoben.")
print(f"  🔁  {skipped_existing} Dateien übersprungen (bereits vorhanden).")
print(f"  🗂️   {skipped_non_images} Nicht-Bilddateien ignoriert.")
for class_name in target_classes:
    class_dir = Path(f"{output_dir}/{class_name}")
    image_files = glob.glob(str(class_dir / "*.jpg"))
    print(f"  {class_name}: {len(image_files)} Dateien")