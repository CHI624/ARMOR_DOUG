import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from tensorflow.keras.models import load_model
from aidrtokenize import tokenize
from pathlib import Path
import sys

# Import central path configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from paths_config import Paths

STATIC_DIR = Paths.DEMO_STATIC

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, "rb") as handle:
        return pickle.load(handle)

def load_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def preprocess_text(text_file, tokenizer, max_seq_length=25):
    with open(text_file, 'r') as f:
        text_data = f.readlines()
    tokenized_texts = [tokenize(line) for line in text_data]
    sequences = tokenizer.texts_to_sequences(tokenized_texts)
    return pad_sequences(sequences, maxlen=max_seq_length, padding='post')

def save_image(image, heatmap):
    # Code to save Grad-CAM image; can be further modularized
    plt.clf()
    plt.matshow(heatmap)
    plt.colorbar()
    plt.savefig(STATIC_DIR / "visualize.jpg")

def load_models(model_paths):
    return {name: load_model(path) for name, path in model_paths.items()}
