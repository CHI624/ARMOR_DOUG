import os
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
from pathlib import Path
import sys

# Shared path config for DMD assets
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from paths_config import Paths  # type: ignore
except Exception:
    Paths = None

# Define paths
base_directory = str(Paths.DMD_MULTIMODAL if Paths else Path("dmd/multimodal"))
output_path = str(Paths.DATA_DUMP / "dmd_images_data_dump_v2.npy") if Paths else "data_dump/dmd_images_data_dump_v2.npy"

# Dictionary to store preprocessed images
images_npy_data = {}

# Function to load and preprocess an image
def preprocess_image(img_path):
    try:
        img = keras_image.load_img(img_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print(f"Processed image {img_path}: shape {x.shape}, dtype {x.dtype}")
        return x
    except Exception as e:
        print(f"Error in preprocessing {img_path}: {e}")
        return None

# Traverse the directory structure and process images
for category in os.listdir(base_directory):
    category_path = os.path.join(base_directory, category)
    images_path = os.path.join(category_path, "images")

    if os.path.isdir(images_path):
        print(f"Scanning directory: {images_path}")
        for file in os.listdir(images_path):
            if file.lower().endswith('.jpg'):  # Check for image files
                img_path = os.path.join(images_path, file)
                try:
                    # Preprocess and add to the dictionary
                    processed_image = preprocess_image(img_path)
                    if processed_image is not None:
                        images_npy_data[img_path] = processed_image
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    else:
        print(f"No valid images directory found under {category_path}.")

# Save the dictionary as a .npy file using pickle
if len(images_npy_data) > 0:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as handle:
        pickle.dump(images_npy_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"DMD image data dump created with {len(images_npy_data)} entries at: {output_path}")
else:
    print("No data to save. Exiting.")
