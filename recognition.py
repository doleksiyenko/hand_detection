import os
from random import shuffle
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential



if __name__ == "__main__":
    if not os.path.isdir('dataset'):
        print('Dataset not found, must have /dataset in current directory')
    
    os.chdir('dataset')
    current_directory = os.getcwd()

    image_height = 144
    image_width = 256

    training_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=current_directory,
        labels='inferred',
        color_mode='grayscale',
        shuffle=True, 
        validation_split=0.2,
        image_size=(image_height, image_width),
        subset='training',
        seed=100
    )

    validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=current_directory,
        labels='inferred',
        color_mode='grayscale',
        shuffle=True, 
        validation_split=0.2,
        image_size=(image_height, image_width),
        subset='validation',
        seed=100
    )
    class_names = training_dataset.class_names

    AUTOTUNE = tf.data.AUTOTUNE

    training_dataset = training_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

    num_classes = 2

    model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(image_height, image_width, 1)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model.summary()

    epochs=3
    history = model.fit(
    training_dataset,
    validation_data=validation_dataset,
    epochs=epochs
    )

    print(history.history)

    model.save('2-gesture-CNN.model')