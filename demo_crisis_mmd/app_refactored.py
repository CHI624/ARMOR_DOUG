from __future__ import division, print_function
import sys
from random import Random
import os
import re
import pickle
import numpy as np
import pandas as pd
import json
import cv2
import aidrtokenize as aidrtokenize
import data_process_multimodal_pair as data_process
from crisis_data_generator_image_optimized import DataGenerator
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model, Model
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras.backend as K
import traceback
from tensorflow.keras.preprocessing import image as keras_image
from flask import Flask, request, render_template, redirect, url_for
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import glob
from pathlib import Path

# Allow importing the shared path configuration from the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
from paths_config import Paths

matplotlib.use('Agg')

app = Flask(
    __name__,
    static_folder=str(Paths.DEMO_STATIC),
    template_folder=str(Paths.DEMO_TEMPLATES),
)

# Base directory for resolving local resources
BASE_DIR = Paths.DEMO_ROOT
DATA_ROOT = os.environ.get("CRISIS_DATA_ROOT", str(BASE_DIR))

# For TESTING
# Predefined list of indices for testing
predefined_indices = [0, 5, 10, 15, 20]
current_index_position = 0  # Global index position

# Create a new Random instance with the same seed for reproducibility
rng = Random(7)

# Shared directories from central config
PERFORMANCE_DIR = Paths.DEMO_PERFORMANCE
MODEL_DIR = Paths.MODEL_DIR
DATA_DUMP_DIR = Paths.DEMO_DATA_DUMP
METADATA_DIR = Paths.DEMO_METADATA
DMD_METADATA_FILE = Paths.DMD_ROOT / "dmd_metadata.csv"

# Load performance measures for CrisisMMD
inf = pd.read_csv(PERFORMANCE_DIR / "informative.csv")
hum = pd.read_csv(PERFORMANCE_DIR / "humanitarian.csv")
sev = pd.read_csv(PERFORMANCE_DIR / "severity.csv")

# Model paths
MODEL_PATHS = {
    "informative": [
        str(MODEL_DIR / "model_info_x.hdf5"),
        str(MODEL_DIR / "model_info_x1.hdf5"),
        str(MODEL_DIR / "model_info_x2.hdf5"),
    ],
    "humanitarian": [
        str(MODEL_DIR / "model_x.hdf5"),
        str(MODEL_DIR / "model_x1.hdf5"),
        str(MODEL_DIR / "model_x2.hdf5"),
    ],
    "severity": [
        str(MODEL_DIR / "model_severe_x.hdf5"),
        str(MODEL_DIR / "model_severe_x1.hdf5"),
        str(MODEL_DIR / "model_severe_x2.hdf5"),
    ],
    "text_models": {
        "informative": str(MODEL_DIR / "informativeness_cnn_keras.hdf5"),
        "humanitarian": str(MODEL_DIR / "humanitarian_cnn_keras_09-04-2022_05-10-03.hdf5"),
        "severity": str(MODEL_DIR / "severity_cnn_keras_21-07-2022_08-14-32.hdf5"),
    },
    "image_models": {
        "informative": str(MODEL_DIR / "informative_image.hdf5"),
        "humanitarian": str(MODEL_DIR / "humanitarian_image_vgg16_ferda.hdf5"),
        "severity": str(MODEL_DIR / "severity_image.hdf5"),
    }
}

# Load images data for CrisisMMD
crisis_mmd_images_npy_data = np.load(DATA_DUMP_DIR / "all_images_data_dump.npy", allow_pickle=True, mmap_mode='r')

# Load images data for DMD if available
dmd_images_npy_data = np.load(DATA_DUMP_DIR / "dmd_images_data_dump.npy", allow_pickle=True, mmap_mode='r')

# Load CrisisMMD metadata
crisis_mmd_metadata_paths = {
    "informative": METADATA_DIR / "task_informative_text_img_agreed_lab_test.tsv",
    "humanitarian": METADATA_DIR / "task_humanitarian_text_img_agreed_lab_test.tsv",
    "severity": METADATA_DIR / "task_severity_test.tsv",
}

crisis_mmd_data_dict = {task: pd.read_csv(path, sep="\t") for task, path in crisis_mmd_metadata_paths.items()}

# Preprocess labels for humanitarian task in CrisisMMD
df1 = crisis_mmd_data_dict["humanitarian"]
df1.loc[df1["label"] == "missing_or_found_people", "label"] = "affected_individuals"
df1.loc[df1["label"] == "injured_or_dead_people", "label"] = "affected_individuals"
df1.loc[df1["label"] == "vehicle_damage", "label"] = "infrastructure_and_utility_damage"

# Extract data for CrisisMMD
crisis_mmd_data = {}
for task in ["informative", "humanitarian", "severity"]:
    df = crisis_mmd_data_dict[task]
    crisis_mmd_data[task] = {
        "images": list(df['image'].values),
        "texts": list(df['tweet_text'].values),
        "labels": list(df['label'].values)
    }


# Load dmd metadata
dmd_metadata_path = DMD_METADATA_FILE

dmd_metadata = pd.read_csv(dmd_metadata_path)

dmd_data = {}
for task in ["informative", "humanitarian", "severity"]:
    crisis_mmd_df = crisis_mmd_data_dict[task]
    
    # Load text content from the text files
    texts = []
    for text_path in dmd_metadata['text_path'].values:
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                texts.append(file.read().strip())
        except Exception as e:
            print(f"Error reading file {text_path}: {e}")
            texts.append("")  # Add empty string for missing or unreadable files


    #Note we still pass the labels in crisis_mmd. This is only for the label encoder to map to the right labels
    dmd_data[task] = {
        "images": list(dmd_metadata['image_path'].values),
        "texts": texts,
        "labels": list(crisis_mmd_df['label'].values)
    }

# We'll load data for DMD in a similar manner
DMD_BASE_PATH = Paths.DMD_MULTIMODAL

# def load_dmd_data():
#     # This function will load images and corresponding text from the DMD dataset
#     dmd_images = []
#     dmd_texts = []
#     dmd_labels = []

#     # Categories in DMD dataset
#     categories = ["damaged_infrastructure", "damaged_nature", "fires", "flood", "human_damage", "non_damage"]
#     for category in categories:
#         category_images_path = os.path.join(DMD_BASE_PATH, category, "images")
#         category_texts_path = os.path.join(DMD_BASE_PATH, category, "text")

#         if os.path.isdir(category_images_path) and os.path.isdir(category_texts_path):
#             image_files = glob.glob(os.path.join(category_images_path, "*.jpg"))
#             text_files = glob.glob(os.path.join(category_texts_path, "*.txt"))

#             text_mapping = {os.path.splitext(os.path.basename(txt_file))[0]: txt_file for txt_file in text_files}

#             for img_file in image_files:
#                 basename = os.path.splitext(os.path.basename(img_file))[0]
#                 txt_file = text_mapping.get(basename)
#                 if txt_file:
#                     dmd_images.append(img_file)
#                     with open(txt_file, "r", encoding="utf-8") as f:
#                         content = f.read()
#                     dmd_texts.append(content)
#                     # Using category as a placeholder label for demonstration
#                     dmd_labels.append(category)
#                 else:
#                     print(f"No corresponding text found for image {img_file} in category {category}")
#         else:
#             print(f"Category directories not found for {category}")

#     return {
#         "images": dmd_images,
#         "texts": dmd_texts,
#         "labels": dmd_labels
#     }

# # Load the DMD data
# dmd_data_full = load_dmd_data()

print('Models will be loaded as needed.')

def preprocess_input_vgg(x):
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

def save_image(image_path, heatmap):
    img = keras_image.load_img(image_path)
    img = keras_image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras_image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras_image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = keras_image.array_to_img(superimposed_img)
    plt.clf()
    plt.matshow(superimposed_img)
    plt.colorbar()
    plt.savefig(Paths.DEMO_STATIC / "visualize.jpg")

def _get_text_xticks(sentence):
    tokens = [word_.strip() for word_ in sentence.split(' ')]
    return tokens

def _plot_score(vec, pred_text, xticks):
    plt.clf()
    fig = plt.figure(figsize=(5, 4))
    plt.yticks([])
    plt.xticks(range(0, len(vec)), xticks, fontsize=15, rotation='vertical')
    img = plt.imshow([vec], vmin=0, vmax=1, origin="lower")
    plt.colorbar()
    plt.savefig(Paths.DEMO_STATIC / "text.jpg")

def load_models(task):
    models = [load_model(path) for path in MODEL_PATHS[task]]
    return models

def run_predictions(dataset_name, data, images_npy_data, task, index, model_index):
    image_file_list = data[task]["images"][index:index+4]
    text_file_list = data[task]["texts"][index:index+4]
    
    # Only CrisisMMD has predefined labels; for DMD, this will be set to None
    labels = data[task]["labels"][index:index+4] if dataset_name == "CrisisMMD" else None
    print("Labels in run_predictions:", labels)
    tokenizer = pickle.load(open(MODEL_DIR / "info_multimodal_paired_agreed_lab.tokenizer", "rb"))

    test_x, test_image_list, test_y, test_le, test_labels = data_process.read_dev_data_multimodal(
        image_file_list, text_file_list, labels, tokenizer, 25, "\t", data[task]["labels"]
    )

    # Create data generator for test images
    params = {
        "max_seq_length": 25,
        "batch_size": 4,
        "n_classes": len(test_labels) if test_labels else 0,
        "shuffle": False
    }
    test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)

    # Load models for the specific task
    models = load_models(task)
    output_preds = [model.predict(test_data_generator, verbose=1) for model in models]

    # Sum up the predictions across models
    summed = np.sum(output_preds, axis=0)
    predicted_indices = np.argmax(summed, axis=1)
    output_labels = [test_labels[idx] for idx in predicted_indices]
    # # If using CrisisMMD, map predictions to predefined labels
    # if dataset_name == "CrisisMMD":
    #     output_labels = [test_labels[idx] for idx in predicted_indices]
    # else:
    #     # For DMD, generate dynamic label names based on predictions
    #     output_labels = [f"predicted_label_{idx}" for idx in predicted_indices]

    # Construct result dictionaries for each model output (for UI display)
    result_dicts = []
    for i in range(len(summed)):
        result = {test_labels[j]: summed[i][j] for j in range(len(test_labels))}
        result_dicts.append(result)

    # Ensure model_index is within range
    if model_index >= len(result_dicts):
        model_index = 0

    selected_result = result_dicts[model_index]

    # Model-specific results for display
    m1 = {test_labels[i]: output_preds[0][model_index][i] for i in range(len(test_labels))}
    m2 = {test_labels[i]: output_preds[1][model_index][i] for i in range(len(test_labels))}
    m3 = {test_labels[i]: output_preds[2][model_index][i] for i in range(len(test_labels))}

    return output_labels, selected_result, m1, m2, m3, test_labels


def get_selected_dataset(selected_dataset):
    # Returns appropriate data dictionary and images array for the selected dataset
    if selected_dataset == "dmd":
        # If user selected DMD dataset
        dataset_name = "DMD"
        data = dmd_data
        images_data = dmd_images_npy_data
    else:
        # Default or user selected CrisisMMD dataset
        dataset_name = "CrisisMMD"
        data = crisis_mmd_data
        images_data = crisis_mmd_images_npy_data
    return data, images_data, dataset_name

@app.route('/', methods=['GET', 'POST'])
def index():
    # Check both form data and query string for dataset selection
    selected_dataset = request.form.get("datasetOption") or request.args.get("dataset", "crisis_mmd")
    
    # Get the correct data and images based on the selected dataset
    data, images_npy_data, dataset_name = get_selected_dataset(selected_dataset)

    # Generate random indices as before for sample display
    random_index1 = rng.randint(0, max(0, len(data["informative"]["images"]) - 4))
    random_index2 = rng.randint(0, max(0, len(data["humanitarian"]["images"]) - 4))
    random_index3 = rng.randint(0, max(0, len(data["severity"]["images"]) - 4))
    
    indices = [random_index1, random_index2, random_index3]
    index = min(indices) if indices else 0

    return render_template(
        'index.html',
        datasetOption=selected_dataset,  # Pass selected dataset to template
        img1=data["informative"]["images"],
        img2=data["humanitarian"]["images"],
        img3=data["severity"]["images"],
        text1=data["informative"]["texts"],
        text2=data["humanitarian"]["texts"],
        text3=data["severity"]["texts"],
        radio=1, m1={}, m2={}, m3={}, l1=0, l2=0, l3=0,
        index=index, output=None, result={}, len=0, labels=[], i="0"
    )

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     selected_dataset = request.form.get("datasetOption", "crisis_mmd")
#     data, images_npy_data, dataset_name = get_selected_dataset(selected_dataset)

#     # Debug: Print lengths of each image and text list
#     print(f"{dataset_name} - Dataset lengths:")
#     for task_key in data.keys():
#         print(f"Length of data['{task_key}']['images']: {len(data[task_key]['images'])}")

#     # Generate random indices using the local RNG instance
#     random_index1 = rng.randint(0, max(0, len(data["informative"]["images"]) - 4))
#     random_index2 = rng.randint(0, max(0, len(data["humanitarian"]["images"]) - 4))
#     random_index3 = rng.randint(0, max(0, len(data["severity"]["images"]) - 4))

#     indices = [random_index1, random_index2, random_index3]
#     indices = [idx for idx in indices if idx >= 0]  # Ensure non-negative
#     index = min(indices) if indices else 0

#     # Debug: Print the selected random indices
#     print(f"Selected random indices for {dataset_name}:")
#     print(f"Random Index 1 (Informative): {random_index1}")
#     print(f"Random Index 2 (Humanitarian): {random_index2}")
#     print(f"Random Index 3 (Severity): {random_index3}")
#     print(f"Final index used for slicing: {index}")

#     # Debug: Print the selected images and texts for display
#     print(f"{dataset_name} - Images and texts selected for display:")
#     for task_key in data.keys():
#         print(f"Images ({task_key}): {data[task_key]['images'][index:index+4]}")
#         print(f"Texts ({task_key}): {data[task_key]['texts'][index:index+4]}")

#     return render_template(
#         'index.html',
#         datasetOption=dataset_name,
#         img1=data["informative"]["images"],
#         img2=data["humanitarian"]["images"],
#         img3=data["severity"]["images"],
#         text1=data["informative"]["texts"],
#         text2=data["humanitarian"]["texts"],
#         text3=data["severity"]["texts"],
#         radio=1, m1={}, m2={}, m3={}, l1=0, l2=0, l3=0,
#         index=index, output=None, result={}, len=0, labels=[], i="0"
#     )

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        selected_dataset = request.form.get("datasetOption", "crisis_mmd")
        print("Request form:", request.form)
        data, images_npy_data, dataset_name = get_selected_dataset(selected_dataset)
        print("Selected dataset in /predict:", selected_dataset)
        task_option = request.form['inlineRadioOptions']
        task_map = {"option1": "informative", "option2": "humanitarian", "option3": "severity"}
        task = task_map.get(task_option, "informative")

        index = int(request.form['index'])
        model_index = int(request.form['index1'])

        output_labels, result, m1, m2, m3, test_labels = run_predictions(dataset_name, data, images_npy_data, task, index, model_index)

        return render_template(
            'index.html',
            datasetOption=dataset_name,
            img1=data["informative"]["images"],
            img2=data["humanitarian"]["images"],
            img3=data["severity"]["images"],
            text1=data["informative"]["texts"],
            text2=data["humanitarian"]["texts"],
            text3=data["severity"]["texts"],
            radio={"informative": 1, "humanitarian": 2, "severity": 3}.get(task, 1),
            m1=m1, m2=m2, m3=m3, l1=len(m1), l2=len(m2), l3=len(m3),
            index=index, output=output_labels, result=result, len=len(result), labels=test_labels, i=model_index
        )

@app.route('/details', methods=['POST'])
def details():
    try:
        print("Entered /details route")

        selected_dataset = request.form.get("datasetOption", "crisis_mmd")
        data, images_npy_data, dataset_name = get_selected_dataset(selected_dataset)

        index = int(request.form.get('index', 0))
        model_index = int(request.form.get('index1', 0))
        task_option = request.form.get('inlineRadioOptions', '')
        task_map = {"option1": "informative", "option2": "humanitarian", "option3": "severity"}
        task = task_map.get(task_option, "informative")

        print(f"{dataset_name} details - index: {index}, model_index: {model_index}, task_option: {task_option}")

        image_file_list = data[task]["images"][index:index+4]
        text_file_list = data[task]["texts"][index:index+4]
        labels = data[task]["labels"][index:index+4]

        # Parse 'selected_result' from form data
        result_string = request.form.get('index2', '{}')
        print(f"Result string: {result_string}")

        # Safely parse the result string into a dictionary
        try:
            selected_result = json.loads(result_string.replace("'", '"'))
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            selected_result = {}
        print(f"Selected result: {selected_result}")

        tokenizer_filenames = {
            "informative": MODEL_DIR / "informativeness_cnn_keras_09-04-2022_04-26-49.tokenizer",
            "humanitarian": MODEL_DIR / "humanitarian_cnn_keras_09-04-2022_05-10-03.tokenizer",
            "severity": MODEL_DIR / "severity_cnn_keras_21-07-2022_08-14-32.tokenizer",
        }
        tokenizer_path = tokenizer_filenames.get(task)
        if not tokenizer_path or not Path(tokenizer_path).is_file():
            error_message = f"Tokenizer file not found: {tokenizer_path}"
            print(error_message)
            return error_message, 500

        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully.")

        test_x, _, test_y, test_le, test_labels = data_process.read_dev_data_multimodal(
            image_file_list, text_file_list, labels, tokenizer, 25, "\t", data[task]["labels"]
        )

        text_model_path = MODEL_PATHS["text_models"][task]
        if not os.path.isfile(text_model_path):
            error_message = f"Text model file not found: {text_model_path}"
            print(error_message)
            return error_message, 500
        text_model = load_model(text_model_path)
        print("Text model loaded successfully.")

        output1 = text_model.predict(test_x, batch_size=128, verbose=1)

        test_images = []
        for img_name in image_file_list:
            if images_npy_data is not None and img_name in images_npy_data:
                image_data = images_npy_data.get(img_name)
                if image_data is not None:
                    if len(image_data.shape) == 4 and image_data.shape[0] == 1:
                        image_data = np.squeeze(image_data, axis=0)
                    if image_data.shape != (224, 224, 3):
                        img = keras_image.array_to_img(image_data)
                        img = img.resize((224, 224))
                        image_data = keras_image.img_to_array(img)
                        image_data = preprocess_input(image_data)
                else:
                    print(f"Image data not found for: {img_name} in {dataset_name} dataset.")
                    image_data = np.zeros((224, 224, 3), dtype=np.float32)
            else:
                # If image data is not found in dictionary or dictionary is None
                # Load from disk if possible
                img_path = img_name if os.path.isfile(img_name) else os.path.join(DATA_ROOT, img_name)
                if not os.path.isfile(img_path):
                    print(f"Image file not found: {img_path}")
                    image_data = np.zeros((224, 224, 3), dtype=np.float32)
                else:
                    img = keras_image.load_img(img_path, target_size=(224, 224))
                    image_data = keras_image.img_to_array(img)
                    image_data = preprocess_input(image_data)
            test_images.append(image_data)

        test_images = np.array(test_images)
        print(f"Image data prepared for details. Shape: {test_images.shape}")

        image_model_path = MODEL_PATHS["image_models"][task]
        if not os.path.isfile(image_model_path):
            error_message = f"Image model file not found: {image_model_path}"
            print(error_message)
            return error_message, 500
        image_model = load_model(image_model_path)
        print("Image model loaded successfully for details.")

        output2 = image_model.predict(test_images, batch_size=128, verbose=1)

        # If `test_labels` is empty (which could happen with DMD if no labels), provide a default
        if not test_labels:
            test_labels = [f'label_{i}' for i in range(len(output1[model_index]))]

        m1 = {test_labels[i]: float(output1[model_index][i]) for i in range(len(test_labels))}
        m2 = {test_labels[i]: float(output2[model_index][i]) for i in range(len(test_labels))}

        output_label_text = test_labels[np.argmax(output1[model_index])] if test_labels else 'unknown'
        output_label_image = test_labels[np.argmax(output2[model_index])] if test_labels else 'unknown'

        performance_data = {"informative": inf, "humanitarian": hum, "severity": sev}.get(task, pd.DataFrame())

        result = selected_result

        print("About to render result.html template from details")

        return render_template(
            'result.html',
            datasetOption=dataset_name,
            result=result,
            m1=m1,
            m2=m2,
            m3=selected_result,
            img=image_file_list,
            text=text_file_list,
            labels=test_labels,
            output1=output_label_image,
            output=output_label_text,
            l1=len(m1),
            l2=len(m2),
            l3=len(selected_result),
            column_names=performance_data.columns.values if not performance_data.empty else [],
            row_data=list(performance_data.values.tolist()) if not performance_data.empty else [],
            model_name=task.capitalize(),
            index=model_index,
            output2=request.form.get('index3', ''),
        )

    except Exception as e:
        print(f"Error in /details route: {e}")
        traceback.print_exc()
        return "An error occurred in the details route.", 500

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        selected_dataset = request.form.get("datasetOption", "crisis_mmd")
        data, images_npy_data, dataset_name = get_selected_dataset(selected_dataset)

        index_form = int(request.form.get('index', 0))
        index1_form = int(request.form.get('index1', 0))
        index = index_form + index1_form

        task_option = request.form.get('inlineRadioOptions', '')
        task_map = {"option1": "informative", "option2": "humanitarian", "option3": "severity"}
        task = task_map.get(task_option, "informative")

        if task is None:
            print("Invalid task option for visualize.")
            return "Invalid task option.", 400

        if index < 0 or index >= len(data[task]["images"]):
            print(f"Index {index} out of range for {task} in {dataset_name} dataset.")
            return "Index out of range.", 400

        if not data[task]["images"]:
            print(f"No images available for task {task} in {dataset_name} dataset.")
            return "No images available for the selected task and dataset.", 500

        image_path = data[task]["images"][index]
        text = data[task]["texts"][index]
        label = data[task]["labels"][index] if index < len(data[task]["labels"]) else "unknown"

        print(f"Visualization for {dataset_name} - Task: {task}, Index: {index}")
        print(f"Image Path: {image_path}")
        print(f"Text: {text}")
        print(f"Label: {label}")

        if not os.path.isfile(image_path):
            # If image path is not found directly, attempt to reconstruct or default path
            image_path_on_disk = os.path.join(DATA_ROOT, image_path)
            if not os.path.isfile(image_path_on_disk):
                print(f"Image file not found for visualization: {image_path}")
                return f"Image file not found for visualization: {image_path}", 500
            image_path = image_path_on_disk

        img = keras_image.load_img(image_path, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print(f"Image shape after preprocessing: {x.shape}")

        tokenizer_path = MODEL_DIR / "info_multimodal_paired_agreed_lab.tokenizer"
        if not tokenizer_path.is_file():
            print(f"Tokenizer file not found: {tokenizer_path}")
            return f"Tokenizer file not found: {tokenizer_path}", 500

        tokenizer = pickle.load(open(tokenizer_path, "rb"))
        print("Tokenizer loaded successfully for visualization.")

        txt = aidrtokenize.tokenize(text)
        sequences = tokenizer.texts_to_sequences([txt])
        flat_sequence = [item for sublist in sequences for item in sublist]
        data_seq = pad_sequences([flat_sequence], maxlen=25, padding='post')
        print(f"Data sequence shape after padding: {data_seq.shape}")

        model = load_models(task)[0]
        print("Model loaded successfully for visualization.")

        last_conv_layer_name = 'block5_conv3'
        try:
            conv_layer = model.get_layer(last_conv_layer_name)
        except ValueError:
            print(f"Layer '{last_conv_layer_name}' not found in the model.")
            return f"Layer '{last_conv_layer_name}' not found in the model.", 500

        heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

        with tf.GradientTape() as gtape:
            conv_output, predictions = heatmap_model([x, data_seq])
            pred_index = tf.argmax(predictions[0])
            loss = predictions[:, pred_index]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + K.epsilon())
        heatmap = heatmap.numpy()
        print(f"Image heatmap shape: {heatmap.shape}")
        save_image(image_path, heatmap)
        print("Image heatmap saved for visualization.")

        # For text Grad-CAM, more complex logic is needed, and it depends on model details.
        # For now, we skip text Grad-CAM to keep minimal changes and focusing on image Grad-CAM.

        return render_template(
            'visualize.html',
            datasetOption=dataset_name,
            image=image_path,
            text=text,
            img1=data["informative"]["images"],
            img2=data["humanitarian"]["images"],
            img3=data["severity"]["images"],
            text1=data["informative"]["texts"],
            text2=data["humanitarian"]["texts"],
            text3=data["severity"]["texts"],
            radio={"informative": 1, "humanitarian": 2, "severity": 3}.get(task, 1),
            m1={}, m2={}, m3={}, l1=0, l2=0, l3=0,
            index=index_form, output=None, result={}, len=0, labels=[], i="0"
        )
    except Exception as e:
        print(f"[ERROR] Error in /visualize route: {e}")
        traceback.print_exc()
        return "An error occurred in the visualize route.", 500

if __name__ == '__main__':
    app.run()
