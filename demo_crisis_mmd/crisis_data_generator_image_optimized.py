import numpy as np
import keras
from tensorflow.keras.preprocessing.image import array_to_img
import warnings
import datetime
import optparse
import os, errno
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow

def preprocess_input_vgg(x):
    """Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.
    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)
    Note we cannot pass keras.applications.vgg16.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.
    Returns a numpy 3darray (the preprocessed image).
    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]


class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, image_file_list, text_vec, image_vec_dict, labels, max_seq_length=20, batch_size=32,
                 n_classes=2, shuffle=False):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.image_file_list = image_file_list
        self.text_vec = text_vec
        self.image_vec_dict = image_vec_dict
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.image_file_list) / float(self.batch_size)))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        #print(" index starts at: "+str(index * self.batch_size) +" ends at: "+str((index + 1) * self.batch_size))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        temp_indexes= [self.image_file_list[k] for k in indexes]
        print(temp_indexes)
        # Generate data
        images_batch, text_batch, y = self.__data_generation(temp_indexes)

        return [images_batch, text_batch], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_file_list))
        #print(" indexes len: "+str(len(self.indexes)))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        text_batch = np.empty((self.batch_size, self.max_seq_length), dtype=int)
        images_batch = np.empty([self.batch_size, 224, 224, 3])

        print("Data generation started")
        print(f"Batch size: {self.batch_size}, Number of classes: {self.n_classes}, Max sequence length: {self.max_seq_length}")
        print(f"Indexes in current batch: {indexes}")

        # Generate data
        for i, index in enumerate(indexes):
            print(f"\nProcessing index {index} (batch position {i})")
            try:
                if i < len(self.image_file_list):
                    image_file_name = str(index)
                    print(f"Image file name: {image_file_name}")
                    
                    if image_file_name in self.image_vec_dict:
                        # Retrieve image vector from dictionary
                        img = self.image_vec_dict[image_file_name]
                        print(f"Found image in image_vec_dict, shape: {img.shape}")
                        
                        # Assign image to images_batch
                        images_batch[i, :, :, :] = img
                        print(f"Assigned image to images_batch[{i}]")
                        
                        # Store class label
                        print(f"Label shape before assignment: {y[i].shape}")
                        y[i] = self.labels[i]
                        print(f"Assigned label to y[{i}]: {y[i]}")
                        
                        # Store text vector
                        print(f"Text vector shape: {self.text_vec[i].shape}")
                        text_batch[i] = self.text_vec[i]
                        print(f"Assigned text vector to text_batch[{i}]: {text_batch[i]}")
                    else:
                        print(f"Image {image_file_name} not found in image_vec_dict. Assigning default zero image.")
                        img = np.zeros((224, 224, 3))
                        images_batch[i, :, :, :] = img
                        print(f"Assigned default image to images_batch[{i}]")
                else:
                    print(f"Index {i} out of range in image_file_list.")
            except Exception as e:
                print("Exception in data generation for index:", index)
                print("Error:", e)

        print("\nAll images loaded for the current batch")
        current_images_batch = preprocess_input_vgg(images_batch)
        print("Images batch after VGG preprocessing")

        # Final shapes and values
        print(f"Final shape of current_images_batch: {current_images_batch.shape}")
        print(f"Final shape of text_batch: {text_batch.shape}")
        print(f"Final shape of y (labels): {y.shape}")
        print("Batch data generation completed")

        return current_images_batch, text_batch, y
