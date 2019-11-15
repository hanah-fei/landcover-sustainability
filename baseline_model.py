import tensorflow as tf
from tensorflow import keras
import ssl
from tensorflow.keras.preprocessing.image import ImageDataGenerator


"""Baseline model. RGB only, Resnet50 architecture."""

TRAIN_DIRECTORY = '../satellite_data/rgb_dataset/train'
DEV_DIRECTORY = '../satellite_data/rgb_dataset/dev'
TEST_DIRECTORY = '../satellite_data/rgb_dataset/test'


ssl._create_default_https_context = ssl._create_unverified_context
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

# Load data and split into train/validation
# Note that there is no test set here!
datagen = ImageDataGenerator(rescale=1./255) # set validation split
train_generator = datagen.flow_from_directory(
    TRAIN_DIRECTORY,
    target_size=(64, 64),
    class_mode='categorical',
    shuffle = True,
    batch_size = 128)
validation_generator = datagen.flow_from_directory(
    DEV_DIRECTORY,
    target_size=(64, 64),
    shuffle = True,
    batch_size=128,
    class_mode='categorical')
test_generator = datagen.flow_from_directory(
    TEST_DIRECTORY,
    target_size=(64, 64),
    batch_size=1, # batch size for test?
    class_mode='categorical')

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.1))(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Train only the top layers
#for layer in base_model.layers:
#  layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(
      train_generator,
      epochs=10,
      verbose=1,
      validation_data = validation_generator)
