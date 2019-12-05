import tensorflow as tf
from tensorflow import keras
import ssl
import preprocess as pre
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""Resnet50 Model using all 13 bands."""

# create the base pre-trained model
ssl._create_default_https_context = ssl._create_unverified_context
base_model = tf.keras.applications.ResNet50(include_top=False)


train_path = '../satellite_data/train/'
dev_path = '../satellite_data/dev/'
test_path = '../satellite_data/test/'

# Load data and split into train/validation
data_dict = pre.create_data_dict(train_path, dev_path, test_path)

# Create input layer to handle 13 channels
dense_input = tf.keras.layers.Input(shape=(64, 64, 13))
dense_input = tf.keras.layers.Conv2D(3, 3, padding='same')(dense_input)
# Modify output layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x) # 10 classes

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(3, 3, padding='same', name = 'new_conv', input_shape=(64, 64, 13)))

for layer_idx in range(1, 176):
  model.add(base_model.layers[layer_idx])
for l in base_model.layers[1:175]:
  model.add(l)



new_model = base_model.layers.pop(0)
newOutputs = new_model(dense_input)
Model(dense_input, newOutputs)
model = tf.keras.Model(dense_input, newOutputs)

mergedModel = tf.keras.Model(dense_filter, base_model.input)
outputs=base_model.layers[-1].output)


model = tf.keras.Model(dense_filter,predictions)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

X_train = data_dict['train']['X_train']
Y_train = data_dict['train']['Y_train']
model.fit(X_train, Y_train)

