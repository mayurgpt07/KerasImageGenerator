import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


datagen = ImageDataGenerator(rescale=1/255, validation_split=0.0)
datagen_test = ImageDataGenerator(rescale=1/255)

train_images =  datagen.flow_from_directory(
        './trainDirectory/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

test_images = datagen_test.flow_from_directory(
        './',
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,
        classes=['test'])
        
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(150, 150, 3), padding = 'same'),
  tf.keras.layers.MaxPooling2D((3, 3), padding = 'same'),
  tf.keras.layers.Dropout(rate = 0.4),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding = 'same'),
  tf.keras.layers.MaxPooling2D((2,2), padding = 'same'),
  tf.keras.layers.Dropout(rate = 0.4),
  tf.keras.layers.BatchNormalization(),  
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(120, activation='softmax')
])#ResNet50(include_top=True, weights=None, pooling = 'max',classes=120)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(train_images, epochs = 5, batch_size = 64)