import random
import zipfile
import pandas as pd
import os
import requests
import shutil
import glob

from tqdm import tqdm
from pathlib import Path
from io import BytesIO
from math import ceil
from icrawler.builtin import BingImageCrawler


def data_exists():
    workspaceFold = Path(__file__).resolve().parent.parent
 
    # Check if the workspace folder exists
    if not workspaceFold.exists():
        raise FileNotFoundError(f"Workspace folder does not exist: {workspaceFold}")
 
    # Check if the data folder exists
    data_dir = workspaceFold / "data" 
    if data_dir.exists():
        return True
    else:
        return False


def download_images():
    workspaceFold = Path(__file__).resolve().parent.parent

    csv_dir =  workspaceFold / "openimages_csv"
    output_dir =  workspaceFold / "data"

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # === Openimages CSV files ===
    download_links = {
        "oidv7-class-descriptions-boxable.csv": "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv",
        "oidv6-train-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
        "validation-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
        "test-annotations-bbox.csv": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
        "train-images-boxable-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv",
        "validation-images-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
        "test-images-with-rotation.csv": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv"
    }

    # === Step 1 Downloading CSV files from OpenImages === 
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

    # === Step 2 loading Label-ID Mapping and Bounding Box === 
    classes_df = pd.read_csv(f"{csv_dir}/oidv7-class-descriptions-boxable.csv", header=None, names=["LabelName", "ClassName"])
    label_map = dict(zip(classes_df.ClassName, classes_df.LabelName))
    target_labels = {cls: label_map[cls] for cls in target_classes}
    print("\n\nGefundene Label-IDs:", target_labels)

    bbox_train = pd.read_csv(f"{csv_dir}/oidv6-train-annotations-bbox.csv")
    bbox_val = pd.read_csv(f"{csv_dir}/validation-annotations-bbox.csv")
    bbox_test = pd.read_csv(f"{csv_dir}/test-annotations-bbox.csv")

    bbox_df = pd.concat([bbox_train, bbox_val, bbox_test], ignore_index=True)
    filtered_bbox = bbox_df[bbox_df['LabelName'].isin(target_labels.values())]

    # === Step 3 load pictur URL from .csv files === 
    meta_train = pd.read_csv(f"{csv_dir}/train-images-boxable-with-rotation.csv", usecols=["ImageID", "OriginalURL"])
    meta_val = pd.read_csv(f"{csv_dir}/validation-images-with-rotation.csv", usecols=["ImageID", "OriginalURL"])
    meta_test = pd.read_csv(f"{csv_dir}/test-images-with-rotation.csv", usecols=["ImageID", "OriginalURL"])

    image_meta = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)
    image_meta = image_meta.dropna()

    for class_name, label_id in target_labels.items():
        class_dir = Path(f"{output_dir}/{class_name}")
        class_dir.mkdir(parents=True, exist_ok=True)

        image_ids = filtered_bbox[filtered_bbox['LabelName'] == label_id]['ImageID'].unique()
        image_ids = image_ids[:max_images_per_class]

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
                continue


    # === Step 4 Download additional pictures and Extract ZIP ===
    zip_url = "https://storage.googleapis.com/cvstock-932a9-h58gl/1a78s8d2ckm3675v728so2.zip"
    axe_dir = workspaceFold/ "data/Axe"

    print(f"🔽 Lade zusätzliches Datenpaket von {zip_url} herunter...")
    try:
        r = requests.get(zip_url, timeout=10)
        if r.status_code == 200:
            with zipfile.ZipFile(BytesIO(r.content)) as zip_ref:
                zip_ref.extractall(axe_dir)
            print("\n\nZIP erfolgreich extrahiert.")
        else:
            print(f"❌ Fehler beim Herunterladen der ZIP-Datei. Status-Code: {r.status_code}")
    except Exception as e:
        print(f"❌ Fehler beim Herunterladen oder Entpacken der ZIP-Datei: {e}")
    
    keyword = [
        "briefcase",
        "briefcase on table",
        "briefcase street"
    ]

    briefcase_dir = workspaceFold / "data/Briefcase"

    crawler = BingImageCrawler(
        storage={'root_dir': briefcase_dir},
        downloader_threads=4,
        feeder_threads=1
    )

    for kw in keyword:
        crawler.crawl(
            keyword=kw,
            max_num = 550,
            filters={
                "type": "photo"
            }
        )

    # === Step 5 move downloaded images to destination folder  ===
    axe_nested_root = workspaceFold / axe_dir / "images.cv_1a78s8d2ckm3675v728so2/data"
    destination = workspaceFold / axe_dir

    moved = 0
    skipped_existing = 0
    skipped_non_images = 0

    print("Verschiebe Bilder...")
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

    # === Step 5 clean up and data overview ===
    extracted_zip_folder = workspaceFold / axe_dir / "images.cv_1a78s8d2ckm3675v728so2"
    try:
        shutil.rmtree(extracted_zip_folder)
    except Exception as e:
        print(f"Fehler beim Löschen von '{extracted_zip_folder}': {e}")

    print(f"\n✅ Zusammenfassung:")
    print(f"  📦  {moved} Bilder verschoben.")
    print(f"  🔁  {skipped_existing} Dateien übersprungen (bereits vorhanden).")
    print(f"  🗂️   {skipped_non_images} Nicht-Bilddateien ignoriert.\n")
    for class_name in target_classes:
        class_dir = Path(f"{output_dir}/{class_name}")
        image_files = glob.glob(str(class_dir / "*.jpg"))
        print(f"  {class_name}: {len(image_files)} Dateien")
    
    output_dir = Path(output_dir) 
    image_exts = [".jpg", ".jpeg", ".png"]
    splits = {"train": 0.75, "test": 0.25}

    # === Step 6 create split directories === 
    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    print("\n📁 Starte Datensatz-Split...")
    for class_dir in output_dir.iterdir():
        if class_dir.is_dir() and class_dir.name not in splits:
            class_name = class_dir.name
            all_images = [f for f in class_dir.glob("*") if f.suffix.lower() in image_exts]
            random.shuffle(all_images)

            total = len(all_images)
            train_cutoff = ceil(total * splits["train"])

            split_map = {
                "train": all_images[:train_cutoff],
                "test": all_images[train_cutoff:]
            }

            for split, files in split_map.items():
                dest_dir = output_dir / split / class_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                for file in files:
                    shutil.move(str(file), str(dest_dir / file.name))

            shutil.rmtree(class_dir)
            print(f"✅ '{class_name}' in train/test aufgeteilt und Original gelöscht.")

    print("\n✅ Dataset split vollständig.")