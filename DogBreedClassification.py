# import pandas as pd
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


datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2, preprocessing_function=preprocess_input)
datagen_test = ImageDataGenerator(rescale=1/255)

train_images =  datagen.flow_from_directory(
        './trainDirectory/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset = 'training')

val_images =  datagen.flow_from_directory(
        './trainDirectory/',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical',
        subset = 'validation')

test_images = datagen_test.flow_from_directory(
        './',
        target_size=(150, 150),
        batch_size=32,
        class_mode=None,
        classes=['test'])

## Resnet model        
# input_tensor = Input(shape=(150,150,3))        
# model_resnet = ResNet50(include_top=False, input_tensor=input_tensor,input_shape=(150,150,3),weights = 'imagenet', classes = 120)

# conv1 = Conv2D(256, (3,3), activation = 'relu')(model_resnet.output)
# pool1 = MaxPooling2D(2,2)(conv1)
# drop1 = Dropout(0.2)(pool1)

# flatten1 = Flatten()(drop1)
# fc2 = Dense(120, activation='softmax')(flatten1)

# model = Model(inputs = input_tensor, outputs = fc2)
# print(model.summary())

## Custom model for identification 
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(256, (3,3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPooling2D((2, 2)),
  tf.keras.layers.Dropout(rate = 0.4),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(256, (3, 3), activation = 'relu'),
  tf.keras.layers.MaxPooling2D((2,2)),
  tf.keras.layers.Dropout(rate = 0.4),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Flatten(),
  ## Add more layers based on your system performance
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(120, activation='softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(train_images, epochs = 40, batch_size = 128, validation_data = val_images)
print(model.summary())

test_generator.reset()

model.evaluate(test_images)
predict = model.predict(test_images, batch_size = 128)

classification = np.argmax(predict, axis = 1)

