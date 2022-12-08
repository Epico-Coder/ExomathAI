from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
from PIL import Image
sys.modules['Image'] = Image

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_data = train.flow_from_directory((Path(__file__).parent / '../data/fire/Training.csv').resolve(),
                                       target_size=(150, 150),
                                       batch_size=32,
                                       class_mode='binary')
test_data = test.flow_from_directory((Path(__file__).parent / '../data/fire/Testing.csv').resolve(),
                                     target_size=(150, 150),
                                     batch_size=32,
                                     class_mode='binary')

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D((3, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Accuracy: 95-96%
model.fit(train_data, epochs=10, validation_data=test_data)

# model.save((Path(__file__).parent / '../models/cnn.h5').resolve())
