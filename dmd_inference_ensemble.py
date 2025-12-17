import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import aidrtokenize  # Ensure this is in your project or installed
from pathlib import Path
from paths_config import Paths

# Set task and input paths here
TASK = "severity"  # Options: "informative", "humanitarian", "severity"
DMD_TEXT_FILE = Paths.DMD_MULTIMODAL / "flood" / "text" / "accrafloods_2015-06-06_19-45-49.txt"
DMD_IMAGE_FILE = Paths.DMD_MULTIMODAL / "flood" / "images" / "accrafloods_2015-06-06_19-45-49.jpg"

# Task-specific configuration
TASK_CONFIG = {
    "informative": {
        "model_paths": [
            Paths.MODEL_DIR / "model_info_x.hdf5",
            Paths.MODEL_DIR / "model_info_x1.hdf5",
            Paths.MODEL_DIR / "model_info_x2.hdf5",
        ],
        "label_mapping": {
            0: "not_informative",
            1: "informative"
        }
    },
    "humanitarian": {
        "model_paths": [
            Paths.MODEL_DIR / "model_x.hdf5",
            Paths.MODEL_DIR / "model_x1.hdf5",
            Paths.MODEL_DIR / "model_x2.hdf5",
        ],
        "label_mapping": {
            0: "affected_individuals",
            1: "infrastructure_and_utility_damage",
            2: "not_humanitarian",
            3: "other_relevant_information",
            4: "rescue_volunteering_or_donation_effort"
        }
    },
    "severity": {
        "model_paths": [
            Paths.MODEL_DIR / "model_severe_x.hdf5",
            Paths.MODEL_DIR / "model_severe_x1.hdf5",
            Paths.MODEL_DIR / "model_severe_x2.hdf5",
        ],
        "label_mapping": {
            0: "little_or_no_damage",
            1: "mild_damage",
            2: "severe_damage"
        }
    }
}

# Load tokenizer
tokenizer_path = Paths.MODEL_DIR / "info_multimodal_paired_agreed_lab.tokenizer"
with open(tokenizer_path, "rb") as handle:
    tokenizer = pickle.load(handle)

def load_and_preprocess_dmd_data(text_file, image_file, max_seq_length=25):
    with open(text_file, 'r') as f:
        text_data = f.read().strip()

    tokenized_text = aidrtokenize.tokenize(text_data)
    sequence = tokenizer.texts_to_sequences([tokenized_text])
    padded_text = pad_sequences(sequence, maxlen=max_seq_length, padding='post')

    img = image.load_img(image_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    return padded_text, img_array

def run_ensemble_inference(text_file, image_file, task):
    if task not in TASK_CONFIG:
        raise ValueError(f"Unsupported task '{task}'. Choose from: {list(TASK_CONFIG.keys())}")

    config = TASK_CONFIG[task]
    text_data, image_data = load_and_preprocess_dmd_data(text_file, image_file)
    label_mapping = config["label_mapping"]
    label_list = [label_mapping[i] for i in range(len(label_mapping))]

    predictions = []
    for i, model_path in enumerate(config["model_paths"]):
        model = load_model(str(model_path))
        pred = model.predict([image_data, text_data], verbose=0)
        predictions.append(pred)

        pred_probs = pred[0]
        pred_label_idx = np.argmax(pred_probs)
        pred_label = label_mapping[pred_label_idx]

        print(f"\n--- Model {i+1} Prediction ---")
        for j, prob in enumerate(pred_probs):
            print(f"{label_mapping[j]}: {prob:.4f}")
        print(f"Predicted Label (Model {i+1}): {pred_label}")

    # Ensemble (sum) predictions
    summed_preds = np.mean(predictions, axis=0)
    ensemble_index = np.argmax(summed_preds, axis=1)[0]
    ensemble_result = {label_mapping[i]: float(summed_preds[0][i]) for i in range(len(label_mapping))}

    print(f"\n{'='*35}")
    print(f"--- Final Ensemble Result ({task.capitalize()} Task) ---")
    print(f"{'='*35}")
    for label, score in ensemble_result.items():
        print(f"{label}: {score:.4f}")
    print(f"\nEnsemble Predicted Label: {label_mapping[ensemble_index]}")
    print(f"{'='*35}\n")

    return ensemble_index, ensemble_result, label_mapping

# Run inference
if __name__ == "__main__":
    run_ensemble_inference(DMD_TEXT_FILE, DMD_IMAGE_FILE, task=TASK)
