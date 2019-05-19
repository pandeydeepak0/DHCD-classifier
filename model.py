import keras 
from keras import layers
from keras import models
from keras.preprocessing import image
from keras import datasets
from keras.layers import Activation, Convolution2D, Conv2D
from keras.layers import SeparableConv2D, Flatten, Dropout, Dense, MaxPool2D
from keras.models import Sequential, save_model
from keras.layers import Input, BatchNormalization
from keras.regularizers import l2


import numpy as np 
import pandas as pd
import cv2
import matplotlib.pyplot as plt 

#input size for image (32*32) with 1 channel(gray image)
image_size= (32, 32, 1)
#define batch size
batch_size= 128
#no of epochs
epochs= 10
#no of classes : 36- letters, 10- numerals
no_classes= 36
#train_sample= len(xtrain)
#test_sample= len(xtest)
path=('/')
'''
#set regularization to avoid overfitting and penalize with l2
#regularization = regularizers.l2(0.01)

#convert data to float and perform nomalization
xtrain= xtrain.astype('float32')/255
xtrain= (xtrain-0.5)*2
xtest= xtest.astype('float32')/255
xtest= (xtest-0.5)*2

#generate image data to avoid overfitting
data_gen= image.ImageDataGenerator(
    horizontal_flip=True,
    samplewise_center=False,
    validation_split= 0.0,
    rotation_range= 20,
    featurewise_std_normalization=False,
    width_shift_range= 1,
    zca_whitening= True,
    zoom_range= 0.1,
    data_format= "channels_last"
)
'''
image_shape= Input(image_size)
#sequential keras model
model= Sequential()

model.add(layers.Conv2D(8, (3, 3), strides= (1, 1), activation= 'relu', input_shape= (32, 32, 1)))
model.add(layers.Conv2D(8, (3, 3), strides= (1, 1), padding='valid', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2), strides=(1, 1), padding='same'))

model.add(layers.Conv2D(16, (3, 3), strides= (1, 1), activation= 'relu', input_shape= (32, 32, 1)))
model.add(layers.Conv2D(16, (3, 3), strides= (1, 1), padding='valid', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D((2, 2), strides=(2, 2), padding='same'))

model.add(layers.Dropout(rate= 0.2))
model.add(layers.Flatten())
model.add(layers.Dense(units= 1150, activation= 'relu'))
model.add(layers.Dropout(rate= 0.2))
model.add(layers.Dense(units= no_classes, activation= 'softmax'))
model.summary()
'''
#train model
trained_model= model.fit_generator(data_gen.flow(xtrain, ytrain, batch_size=batch_size),
                                   steps_per_epoch= len(xtrain)/batch_size,
                                   epochs= epochs, validation_data= (xtest, ytest))
