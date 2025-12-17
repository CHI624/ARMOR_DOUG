import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
from pathlib import Path
import sys

try:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from paths_config import Paths  # type: ignore
except Exception:
    Paths = None

# Load a sample image path from your dataset
image_path = "data_image/california_wildfires/10_10_2017/917791044158185473_0.jpg"

# Load the image with target size used in preprocessing
img = keras_image.load_img(image_path, target_size=(224, 224))
x = keras_image.img_to_array(img)  # Convert to numpy array
x = np.expand_dims(x, axis=0)  # Add batch dimension

# Apply preprocess_input from Keras's VGG
processed_img = preprocess_input(x)

# Load the .npy dump for comparison
dump_path = Paths.DATA_DUMP / "all_images_data_dump.npy" if Paths else Path("data_dump/all_images_data_dump.npy")
with open(dump_path, 'rb') as handle:
    images_npy_data = pickle.load(handle)

# Retrieve the corresponding preprocessed image data from the .npy dump
npy_image_data = images_npy_data.get(image_path)

# Verify shapes match
if npy_image_data is not None:
    print("Shapes match:", processed_img.shape == npy_image_data.shape)

    # Calculate differences
    difference = np.abs(processed_img - npy_image_data)
    max_diff = np.max(difference)
    mean_diff = np.mean(difference)

    # Print verification statistics
    print(f"Max difference between processed image and .npy dump: {max_diff}")
    print(f"Mean difference between processed image and .npy dump: {mean_diff}")

    # Optionally print a few pixel values to compare
    print("Sample values from processed image:")
    print(processed_img[0, :5, :5, :])  # Print top-left corner values
    print("Sample values from .npy dump:")
    print(npy_image_data[0, :5, :5, :])  # Print top-left corner values
else:
    print("Image not found in .npy dump.")
