import os
import glob
import csv
from pathlib import Path
import sys

# Import shared directory configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from paths_config import Paths

DMD_BASE_PATH = Paths.DMD_MULTIMODAL
OUTPUT_METADATA_FILE = Paths.DMD_ROOT / "dmd_metadata.csv"

def create_dmd_metadata():
    # List to store metadata rows
    metadata = []

    # Categories in DMD dataset
    categories = ["damaged_infrastructure", "damaged_nature", "fires", "flood", "human_damage", "non_damage"]
    
    for category in categories:
        category_images_path = DMD_BASE_PATH / category / "images"
        category_texts_path = DMD_BASE_PATH / category / "text"

        # Ensure both directories exist
        if os.path.isdir(category_images_path) and os.path.isdir(category_texts_path):
            # Get all image and text files
            image_files = glob.glob(str(category_images_path / "*.jpg"))
            text_files = glob.glob(str(category_texts_path / "*.txt"))

            # Map text files by basename for easy lookup
            text_mapping = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in text_files}

            # Iterate over image files and find matching text files
            for img_file in image_files:
                basename = os.path.splitext(os.path.basename(img_file))[0]
                txt_file = text_mapping.get(basename)

                if txt_file:
                    metadata.append({
                        "image_path": img_file,
                        "text_path": txt_file,
                        "label": category
                    })
                else:
                    print(f"No corresponding text found for image {img_file} in category {category}")
        else:
            print(f"Category directories not found for {category}")

    # Write metadata to a CSV file
    with open(OUTPUT_METADATA_FILE, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "text_path", "label"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"Metadata file created at {OUTPUT_METADATA_FILE}")

if __name__ == "__main__":
    create_dmd_metadata()
