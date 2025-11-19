# -*- coding: utf-8 -*-

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json

batch_size = 6 # 32

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'Dataset/',  # This is the source directory for training images
    target_size=(200, 200),  # All images will be resized to 200 x 200
    batch_size=batch_size,
    # Specify the classes explicitly
    classes=['Fake','Real'],
    # Since we use categorical_crossentropy loss, we need categorical labels
    class_mode='categorical')

import tensorflow as tf

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 200x 200 with 3 bytes color
    # The first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 128 neuron in the fully-connected layer
    tf.keras.layers.Dense(128, activation='relu'),
    # 2 output neurons for 2 classes with the softmax activation
    tf.keras.layers.Dense(2, activation='softmax')
])

model.summary()

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

total_sample = train_generator.n

n_epochs = 6  # 32
epoch_array = []
history = model.fit_generator(
    train_generator,
    steps_per_epoch=int(total_sample / batch_size),
    epochs=n_epochs,
    verbose=1)

print(model.history.history['loss'])
print(model.history.history['acc'])

import os, os.path

main_path = "Dataset/"

d_type = ['Fake','Real']
dataset_length = []
# path joining version for other paths
DIR = main_path + 'Fake'
Fake = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
dataset_length.append(Fake)

DIR = main_path + 'Real'
Real = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
dataset_length.append(Real)

#DIR = main_path + 'Water'
#Water = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])#
#dataset_length.append(Water)

import matplotlib.pyplot as plt

plt.bar(d_type, dataset_length)
plt.title('Dataset Size')
plt.xlabel('DeepFake_Grading')
plt.ylabel('Size')
plt.show()

import matplotlib.pyplot as plt
import numpy as np

epoch_array = []
for i in range(n_epochs):
    epoch_array.append(i)

print(epoch_array)
loss = model.history.history['loss']
acc = model.history.history['acc']

plt.plot(epoch_array, loss, "-o", label="Epoch")
plt.plot(loss, epoch_array, "-o", label="Loss")
plt.legend()
plt.show()

# acc=model.history.history['acc']
plt.plot(epoch_array, acc, "-o", label="Epoch")
plt.plot(acc, epoch_array, "-o", label="Accuraccy")
plt.legend()
plt.show()

model.save('model.h5')
