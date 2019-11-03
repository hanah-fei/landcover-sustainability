import numpy as np
import tifffile as tiff
import os
import tensorflow as tf
from tensorflow import keras

"""Read tif files and convert to Keras friendly numpy arrays.
  
   returns data_dict with keys:
      'train' (dict) with keys: 
          'X_train' (np.array) train data with shape (21600, 64, 64, 13)
          'Y_train' (np.array) train labels with shape (21600, 10) 
      'dev' (dict) with keys:
          'X_dev' (np.array) dev data with shape (2700, 64, 64, 13)
          'Y_dev' (np.array) dev labels with shape (2700, 10)
      'train' (dict) with keys: 
          'X_test' (np.array) test data with shape (2700, 64, 64, 13)
          'Y_test' (np.array) test labels with shape (2700, 10)
"""

train_path = '../satellite_data/train/'
dev_path = '../satellite_data/dev/'
test_path = '../satellite_data/test/'


def create_data_dict(train_path, dev_path, test_path):
  label_dict, reverse_dict = create_label_mapping()
  X_train, Y_train = get_data(train_path, label_dict) 
  X_dev, Y_dev = get_data(dev_path, label_dict)
  X_test, Y_test = get_data(test_path, label_dict)
  data_dict = {}
  data_dict['train'] = {'X_train': X_train, 'Y_train': Y_train}  
  data_dict['dev'] = {'X_dev': X_dev, 'Y_dev': Y_dev}
  data_dict['test'] = {'X_test': X_test, 'Y_test': Y_test}
  return data_dict

def create_label_mapping():
  """Create dicts mapping string label to numeric label and vice versa.
    Returns:
      label_dict: (dict) mapping of string label to numeric label
      reverse_dict: (dict) mapping of numeric label to string label
  """
  labels = ['Forest', 'River', 'Highway', 'AnnualCrop', 'SeaLake', 'HerbaceousVegetation', 'Industrial', 'Residential', 'PermanentCrop', 'Pasture'] 
  label_dict = dict(zip(labels, np.arange(len(labels))))
  reverse_dict = dict(map(reversed, label_dict.items()))
  return label_dict, reverse_dict


def get_data(data_path, label_dict):
  X = []
  Y = []
  filenames = os.listdir(data_path)
  for idx,name in enumerate(filenames):
    image_data = tiff.imread(data_path + name)
    X.append(image_data)
    label = name.split('_')[0]
    label_idx = label_dict[label]
    Y.append(label_idx)
  X = np.asarray(X)
  Y = tf.keras.utils.to_categorical(Y)
  return X, Y
