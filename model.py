# -*- coding: utf-8 -*-

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Define dataset directories
data_dir = "D:/A DATA/Project Module 2021-22/A PYTHON New/DeepFake Face/DeepFake_Face_Complete/dataset"  # Update with actual path
train_dir = data_dir + "/Train"
val_dir = data_dir + "/Validation"
test_dir = data_dir + "/Test"

# Image data preprocessing
image_size = (150, 150)
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(train_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')
val_data = datagen.flow_from_directory(val_dir, target_size=image_size, batch_size=batch_size, class_mode='binary')
test_data = datagen.flow_from_directory(test_dir, target_size=image_size, batch_size=batch_size, class_mode='binary', shuffle=False)

# Build a simple CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(128, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Evaluate the model
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy:.4f}")

# Plot training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model
model.save("face_classification_model.h5")
