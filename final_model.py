import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau
import ssl
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
import numpy as np

"""Final model. RGB only, Resnet50 architecture."""

TRAIN_DIRECTORY = '../satellite_data/rgb_dataset/train'
DEV_DIRECTORY = '../satellite_data/rgb_dataset/dev'
TEST_DIRECTORY = '../satellite_data/rgb_dataset/test'


ssl._create_default_https_context = ssl._create_unverified_context
base_model = tf.keras.applications.ResNet50(include_top=False)

# Load data and split into train/validation/test
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    TRAIN_DIRECTORY,
    target_size=(64, 64),
    class_mode='categorical',
    shuffle = True,
    batch_size = 64)
validation_generator = datagen.flow_from_directory(
    DEV_DIRECTORY,
    target_size=(64, 64),
    shuffle = True,
    batch_size=128,
    class_mode='categorical')
test_generator = datagen.flow_from_directory(
    TEST_DIRECTORY,
    target_size=(64, 64),
    batch_size=1,
    shuffle = False,
    class_mode='categorical')

# Set up callbacks
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                          write_graph=True, write_images=False)
call = ReduceLROnPlateau(monitor='val_loss')

# Define model
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model
history = model.fit_generator(
      train_generator,
      epochs=20,
      verbose=1,
      validation_data = validation_generator,
      callbacks=[tensorboard, call])

# Compute global metrics on validation set
val_generator = datagen.flow_from_directory(
    DEV_DIRECTORY,
    target_size=(64, 64),
    shuffle = False,
    batch_size=1,
    class_mode='categorical')
val_loss, val_accuracy = model.evaluate_generator(val_generator)
val_probabilities = model.predict_generator(val_generator)
val_predictions = np.argmax(val_probabilities, axis = 1)
val_labels = val_generator.classes
val_classes = list(val_generator.class_indices.keys())
print(confusion_matrix(val_labels, val_predictions))
print(classification_report(val_labels, val_predictions, target_names=val_classes))

# Compute metrics on test set
test_loss, test_accuracy = model.evaluate_generator(test_generator)
test_probabilities = model.predict_generator(test_generator)
test_predictions = np.argmax(test_probabilities, axis = 1)
test_labels = test_generator.classes
test_classes = list(test_generator.class_indices.keys())
print(confusion_matrix(test_labels, test_predictions))
print(classification_report(test_labels, test_predictions, target_names=test_classes))

# Export model
model.save('saved_model', save_format='tf')



