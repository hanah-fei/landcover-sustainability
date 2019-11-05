#!/usr/bin/env python
# coding: utf-8

# ### Added Pretrained ImageNet weights

# In[4]:


import tensorflow as tf
from tensorflow import keras
import ssl
#import preprocess as pre
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model



"""Baseline model. RGB only, Resnet50 architecture."""

DATA_DIRECTORY = '/Users/rubirodriguez/Documents/Documents/CS230DeepLearning/Project/satellite_data_rgb/'


ssl._create_default_https_context = ssl._create_unverified_context
base_model = tf.keras.applications.ResNet50(weights = 'imagenet',include_top=False)

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

# add a global spatial average pooling layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# add a fully-connected layer
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)

# add a logistic layer for 10 classes
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

# this is the model we will train
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit_generator(
      train_generator,
      epochs=10,
      verbose=1,
      validation_data = validation_generator)


# ### Trying to train some layers

# #### https://keras.io/applications/#usage-examples-for-image-classification-models

# In[5]:


from keras.applications.inception_v3 import InceptionV3
import tensorflow as tf
from tensorflow import keras
import ssl
#import preprocess as pre
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import backend as K


"""Baseline model. RGB only, Resnet50 architecture."""

DATA_DIRECTORY = '/Users/rubirodriguez/Documents/Documents/CS230DeepLearning/Project/satellite_data_rgb/'


ssl._create_default_https_context = ssl._create_unverified_context
base_model =keras.applications.inception_v3.InceptionV3(weights = 'imagenet',include_top=False)


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

# add a global spatial average pooling layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# add a fully-connected layer
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)

# add a logistic layer for 10 classes
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

# this is the model we will train
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit_generator(
      train_generator,
      epochs=10,
      verbose=1,
      validation_data = validation_generator)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
history = model.fit_generator(
      train_generator,
      epochs=10,
      verbose=1,
      validation_data = validation_generator)

