import os
import glob
import sys
from pathlib import Path

try:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from paths_config import Paths  # type: ignore
except Exception:
    Paths = None

# Define the base path for the DMD dataset
DMD_BASE_PATH = Paths.DMD_MULTIMODAL if Paths else Path("../dmd/multimodal")

# Define categories in the DMD dataset
categories = ["damaged_infrastructure", "damaged_nature", "fires", "flood", "human_damage", "non_damage"]

def cleanup_dmd():
    for category in categories:
        images_path = os.path.join(DMD_BASE_PATH, category, "images")
        texts_path = os.path.join(DMD_BASE_PATH, category, "text")

        if not os.path.isdir(images_path) or not os.path.isdir(texts_path):
            print(f"Warning: Missing directory for category '{category}'. Skipping...")
            continue

        # Get all image and text filenames (without extensions)
        image_files = {os.path.splitext(os.path.basename(img))[0] for img in glob.glob(os.path.join(images_path, "*.jpg"))}
        text_files = {os.path.splitext(os.path.basename(txt))[0] for txt in glob.glob(os.path.join(texts_path, "*.txt"))}

        # Find unmatched images and texts
        unmatched_images = image_files - text_files
        unmatched_texts = text_files - image_files

        # Delete unmatched images
        for img_name in unmatched_images:
            img_path = os.path.join(images_path, f"{img_name}.jpg")
            try:
                os.remove(img_path)
                print(f"Deleted unmatched image: {img_path}")
            except OSError as e:
                print(f"Error deleting image {img_path}: {e}")

        # Delete unmatched text files
        for txt_name in unmatched_texts:
            txt_path = os.path.join(texts_path, f"{txt_name}.txt")
            try:
                os.remove(txt_path)
                print(f"Deleted unmatched text file: {txt_path}")
            except OSError as e:
                print(f"Error deleting text file {txt_path}: {e}")

    print("Cleanup complete.")

if __name__ == "__main__":
    cleanup_dmd()
