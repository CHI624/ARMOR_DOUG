import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import aidrtokenize  # Ensure this is accessible
from pathlib import Path
import sys

# Import shared directory configuration
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from paths_config import Paths

# Load your models
MODEL_PATH = Paths.MODEL_DIR / "model_x2.hdf5"  # Adjust the path as needed
model = load_model(str(MODEL_PATH))

# Load the tokenizer
with open(Paths.MODEL_DIR / "info_multimodal_paired_agreed_lab.tokenizer", "rb") as handle:
    tokenizer = pickle.load(handle)

def load_and_preprocess_dmd_data(dmd_text_file, dmd_image_file, max_seq_length=25):
    # Load and preprocess text data
    with open(dmd_text_file, 'r') as f:
        text_data = f.readlines()  # Adjust if data is structured differently

    tokenized_texts = [aidrtokenize.tokenize(line) for line in text_data]
    sequences = tokenizer.texts_to_sequences(tokenized_texts)
    padded_texts = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

    # Load and preprocess image data
    img = image.load_img(dmd_image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return padded_texts, img_array

def run_inference(dmd_text_file, dmd_image_file):
    # Load and preprocess DMD data
    text_data, image_data = load_and_preprocess_dmd_data(dmd_text_file, dmd_image_file)

    # Run inference (assuming the model takes [image, text] inputs)
    predictions = model.predict([image_data, text_data])
    return predictions

if __name__ == "__main__":
    # Paths to your DMD text and image files
    dmd_text_file = Paths.DMD_MULTIMODAL / "flood" / "text" / "accrafloods_2015-06-06_19-45-49.txt"
    dmd_image_file = Paths.DMD_MULTIMODAL / "flood" / "images" / "accrafloods_2015-06-06_19-45-49.jpg"

    # Run inference
    predictions = run_inference(dmd_text_file, dmd_image_file)
    print("Predictions:", predictions)

    # Define mappings for each task based on provided text labels

    # Task 1: Informative/Not Informative Classification
    TASK_1_MAPPING = {
        0: "informative",
        1: "not_informative"
    }

    # Task 2: Humanitarian Task Classification
    TASK_2_MAPPING = {
        0: "affected_individuals",
        1: "infrastructure_and_utility_damage",
        2: "not_humanitarian",
        3: "other_relevant_information",
        4: "rescue_volunteering_or_donation_effort"
    }

    # Task 3: Severity Assessment
    TASK_3_MAPPING = {
        0: "little_or_no_damage",
        1: "mild_damage",
        2: "severe_damage"
    }

    predicted_labels = np.argmax(predictions, axis=1)
    text_labels = TASK_2_MAPPING.keys()
    print("Text Labels:", text_labels)
    print("Predicted Labels:", predicted_labels)
    print("Text Class:", TASK_2_MAPPING[predicted_labels[0]])
    
