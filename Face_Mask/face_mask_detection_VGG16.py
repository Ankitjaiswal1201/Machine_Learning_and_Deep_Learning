# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:17:21 2020

@author: ankit
"""


import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from IPython.display import display 
from PIL import Image

#input_dataset
train_path = 'dataset/training_set'
test_path = 'dataset/test_set'
validation_path = 'dataset/validation_set'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes = ['mask','nomask'], batch_size = 100)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes = ['mask','nomask'], batch_size = 50)
validation_batches = ImageDataGenerator().flow_from_directory(validation_path, target_size=(224,224), classes = ['mask','nomask'], batch_size = 10)

# model = Sequential([
#         Conv2D(32, (3,3), activation = 'relu', input_shape = (224,224,3)),
#         Flatten(),
#         Dense(2, activation='softmax'),
#         ])

# model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy',metrics=['accuracy'])

# model.fit_generator(train_batches, steps_per_epoch = 20, validation_data = test_batches,
#                     validation_steps = 20,epochs = 10, verbose = 2)

#importing VGG16 model
vgg16_model = keras.applications.vgg16.VGG16()

vgg16_model.summary()

type(vgg16_model)

#Converting model to sequential type
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)



model.summary()

for layer in model.layers:
    layer.trainable = False
    
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batches, steps_per_epoch = 20, validation_data = test_batches,
                    validation_steps = 10,epochs = 10, verbose = 2)

valid_imgs, valid_labels = next(validation_batches)

valid_labels = valid_labels[:,0]

print(valid_labels)

predictions= model.predict_generator(validation_batches, steps=1, verbose = 0)

print(predictions)

print(confusion_matrix(valid_labels,predictions[:,0]))

