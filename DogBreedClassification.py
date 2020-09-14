import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import vgg16
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


datagen = ImageDataGenerator(rescale=1/255, validation_split=0.3, preprocessing_function=preprocess_input)
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
model_resnet = ResNet50(include_top=False, input_tensor=input_tensor,input_shape=(224,224,3) ,weights = None, classes=120)

conv1 = Conv2D(256, (3,3), activation = 'relu')(model_resnet.output)
pool1 = MaxPooling2D(2,2)(conv1)
drop1 = Dropout(0.2)(pool1)

flatten1 = Flatten()(drop1)
fc2 = Dense(120, activation='softmax')(flatten1)

model = Model(inputs=model_resnet.input, outputs=fc2)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(train_images, epochs = 20, batch_size = 64)
# print(model.summary())
# model.fit_generator(
#     train_images,
#     steps_per_epoch = 700,
#     validation_data = val_images, 
#     validation_steps = 300,
#     epochs = 20)