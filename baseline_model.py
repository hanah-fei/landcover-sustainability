import tensorflow as tf
from tensorflow import keras
import ssl
import preprocess as pre
from tensorflow.keras.preprocessing.image import ImageDataGenerator


"""Baseline model. RGB only, Resnet50 architecture."""

DATA_DIRECTORY = '../satellite_data/rgb'


ssl._create_default_https_context = ssl._create_unverified_context
base_model = tf.keras.applications.ResNet50(include_top=False)

# Load data and split into train/validation
# Note that there is no test set here!
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) # set validation split
train_generator = datagen.flow_from_directory(
    DATA_DIRECTORY,
    target_size=(64, 64),
    class_mode='categorical',
    shuffle = True,
    batch_size = 128,
    subset='training') # set as training data
validation_generator = datagen.flow_from_directory(
    DATA_DIRECTORY,
    target_size=(64, 64),
    batch_size=128,
    class_mode='categorical',
    subset='validation') # set as validation data


x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Train only the top layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,
      epochs=4,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)
