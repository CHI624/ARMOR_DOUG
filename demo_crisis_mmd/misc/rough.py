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

# Load the .npy file with `allow_pickle=True` in case it contains non-array structures
data_dump_path = Paths.DATA_DUMP / "all_images_data_dump.npy" if Paths else Path("data_dump/all_images_data_dump.npy")
data = np.load(data_dump_path, allow_pickle=True)

# Check if `data` is a dictionary
if isinstance(data, dict):
    # Print the first few keys and corresponding value types and shapes if applicable
    print("Structure of .npy file:")
    for i, (key, value) in enumerate(data.items()):
        value_info = f"Type: {type(value)}, Shape: {value.shape}" if isinstance(value, np.ndarray) else f"Type: {type(value)}"
        print(f"Key: {key}, {value_info}")
        if i >= 4:  # Limit to the first 5 entries
            break
else:
    # If it's not a dictionary, assume it's an array and show basic info
    print("Array shape:", data.shape)
    print("First few entries:", data[:5] if data.size > 5 else data)
