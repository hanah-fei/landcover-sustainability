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
# Note that there is no test set here!
data_dict = pre.create_data_dict(train_path, dev_path, test_path)

# Create input layer to handle 13 channels
dense_input = tf.keras.layers.Input(shape=(64, 64, 13))
dense_filter = tf.keras.layers.Conv2D(3, 3, padding='same')(dense_input)
# Modify output layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x) # 10 classes

model = tf.keras.Model(dense_input,predictions)

# train only the top layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

X_train = data_dict['train']['X_train']
Y_train = data_dict['train']['Y_train']
model.fit(X_train, Y_train)

