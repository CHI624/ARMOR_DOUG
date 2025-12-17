import numpy as np
from pathlib import Path
import sys

try:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))
    from paths_config import Paths  # type: ignore
except Exception:
    Paths = None

# Load the file directly as a dictionary
dmd_dump_path = Paths.DATA_DUMP / "dmd_images_data_dump.npy" if Paths else Path("data_dump/dmd_images_data_dump.npy")
all_images_data = np.load(dmd_dump_path, allow_pickle=True)

# Verify that it's a dictionary and inspect its contents
print(type(all_images_data))      # Should confirm <class 'dict'>
print(len(all_images_data))       # Number of images if it's a dictionary
print(list(all_images_data.keys())[:5])  # Sample keys, expected to be image file names or paths

# Check the structure of a sample image
sample_image = all_images_data[next(iter(all_images_data))]
print(type(sample_image))         # Expected: <class 'numpy.ndarray'>
print(sample_image.shape)         # Expected shape (e.g., (224, 224, 3))
print(sample_image.dtype)         # Expected data type (e.g., float32 or uint8)


print("===============Single Image============")
# Select a sample image from the dictionary
sample_image_path = list(all_images_data.keys())[0]  # Get the first key
sample_image = all_images_data[sample_image_path]  # Retrieve the image data

# Print some statistics to check the normalization
print(f"Image path: {sample_image_path}")
print(f"Image shape: {sample_image.shape}")
print(f"Data type: {sample_image.dtype}")
print(f"Min pixel value: {sample_image.min()}")
print(f"Max pixel value: {sample_image.max()}")
print(f"Mean pixel value: {sample_image.mean()}")
print(f"Standard deviation of pixel values: {sample_image.std()}")

# Inspect a few pixel values
print("Sample pixel values:")
print(sample_image[0, :5, :5, :])  # Print top-left corner of the image for RGB channels
