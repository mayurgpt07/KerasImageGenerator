import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


datagen = ImageDataGenerator(rescale=1/255, validation_split=0.3)
datagen_test = ImageDataGenerator(rescale=1/255)

train_images =  datagen.flow_from_directory(
        './trainDirectory/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset = 'training')

val_images =  datagen.flow_from_directory(
        './trainDirectory/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset = 'validation')

test_images = datagen_test.flow_from_directory(
        './',
        target_size=(224, 224),
        batch_size=32,
        class_mode=None,
        classes=['test'])
        
input_tensor = Input(shape=(224,224,3))        
model = ResNet50(include_top=False, input_tensor = input_tensor, input_shape = (224,224,3), weights = 'imagenet', classes=120)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
# model.fit(train_images, epochs = 20, batch_size = 64)

model.fit_generator(
    train_images,
    steps_per_epoch = 700,
    validation_data = val_images, 
    validation_steps = 300,
    epochs = 20)