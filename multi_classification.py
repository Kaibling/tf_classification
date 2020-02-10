from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import cv2
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
import pprint
from google.colab import drive



print(tf.__version__)


def show_model_history(histoy,epochs):
  acc = history.history['acc']
  loss = history.history['loss']

  epochs_range = range(epochs)

  plt.figure(figsize=(8, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.show()

batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150


_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
print(validation_dir)

train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures


datagen = ImageDataGenerator(
 #rotation_range=20,
 #width_shift_range=0.2,
 #height_shift_range=0.2,
 horizontal_flip=True
 #zoom_range=0.5
 )

validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

print("[INFO] generating trainings_data ...")
train_data_gen = datagen.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     color_mode="rgb",
                                                     #class_mode="categorical")
                                                     class_mode="sparse")
print("[INFO] generating trainings_data completed")

print("[INFO] generating validation data ...")
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              color_mode="rgb",
                                                              #class_mode="categorical")
                                                              class_mode="sparse")
print("[INFO] generating validation data completed")


total_train = train_data_gen.samples
total_val = val_data_gen.samples
class_names = {v: k for k, v in train_data_gen.class_indices.items()}

print("[INFO] Trainings and Validation Data generation completed")

sample_training_images, _ = next(train_data_gen)

plt.figure()
plt.imshow(sample_training_images[1].astype('uint8'))

plt.colorbar()
plt.grid(False)
plt.show()

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(10,10))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img.astype('uint8'))
        ax.axis('off')
    plt.tight_layout()
    plt.show()

plotImages(sample_training_images[:5])


model = tf.keras.Sequential(
  [
    tf.keras.layers.Reshape(input_shape=(IMG_HEIGHT,IMG_WIDTH,3), target_shape=(IMG_HEIGHT, IMG_WIDTH, 3), name="image"),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)), 
    tf.keras.layers.Activation('relu'), 
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(2, activation='softmax')
  ])

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print("[INFO] Model compiled")


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
  )

show_model_history(history,epochs)

sample_training_images, _ = next(val_data_gen)
predictions = model.predict(sample_training_images)
for a in range(6):
  nu = random.randint(0, len(sample_training_images))
  print("nr "+str(nu) + " is a " + class_names[np.argmax(predictions[nu])])
  print(predictions[nu])
  plt.figure()
  plt.imshow(sample_training_images[nu].astype('uint8'))
  plt.imshow(sample_training_images[nu])
  plt.colorbar()
  plt.grid(False)
  plt.show()