import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import roc_curve
from pprint import pprint
import json
tf.enable_eager_execution()

'''Make predictions on new Sentinel2 data.'''

'''
LABELS

FOREST: 0
SEALAKE: 1
ANNUALCROP: 2
PASTURE: 3
INDUSTRIAL: 4
RESIDENTIAL: 5
RIVER: 6
HERBACEOUSVEGETATION: 7
HIGHWAY: 8
PERMANENTCROP: 9

'''



# Load model
model = keras.models.load_model('saved_model')


def config():
  '''Get list of tfrecord files.'''
  files_list = !gsutil ls 'gs://sentinel_true_color'
  export_files_list = [s for s in files_list if  'sentinel_2_rgb' in s]
  image_files_list = []
  json_file = None
  for f in export_files_list:
      if f.endswith('.tfrecord.gz'):
          image_files_list.append(f)
      elif f.endswith('.json'):
          json_file = f
  image_files_list.sort()
  pprint(image_files_list)
  print(json_file)
  return image_files_list


def parse_function(proto):
  '''Parse a single record.'''
  BANDS = ['TCI_R', 'TCI_G', 'TCI_B']
  image_columns = [tf.io.FixedLenFeature(shape=[64,64], dtype=tf.float32) for k in BANDS]
  image_features_dict = dict(zip(BANDS, image_columns))
  parsed_features = tf.parse_single_example(proto, image_features_dict)
  return parsed_features

def to_tuple(inputs):
  '''Preprocess the band data.'''
  inputsList = [inputs.get(key) for key in BANDS]
  stacked = tf.stack(inputsList, axis=0)
  stacked = tf.transpose(stacked, [1, 2, 0])
  stacked = stacked * 10000 # Band data is between 0 and 0.0255 
  stacked = stacked/225
  return stacked[:,:,:len(BANDS)], stacked[:,:,len(BANDS):]

def create_dataset(image_files_list):
  dataset =  tf.data.TFRecordDataset(image_files_list, compression_type='GZIP')
  dataset = dataset.map(parse_function)
  dataset = dataset.map(to_tuple).batch(20)
  return dataset


parsed_records = create_dataset(image_files_list)
probabilities = model.predict_generator(parsed_records)
predictions = np.argmax(probabilities, axis = 1)
