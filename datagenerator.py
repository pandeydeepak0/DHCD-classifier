import keras
from keras.preprocessing.image import ImageDataGenerator
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

base_path=('/')
#create a constructor for datagenerator class
datagen = ImageDataGenerator(
    horizontal_flip=True,
    samplewise_center=False,
    validation_split= 0.0,
    rotation_range= 20,
    featurewise_std_normalization=False,
    width_shift_range= 1,
    zca_whitening= True,
    zoom_range= 0.1,
    data_format= "channels_last")
#fetch the trainig data batch-wise
train_data= datagen.flow_from_directory('Data/Train', color_mode= 'grayscale',target_size= (32, 32), batch_size= 32, class_mode='categorical')
validation_data= datagen.flow_from_directory('Data/Validation', color_mode= 'grayscale', target_size= (32, 32), batch_size= 32, class_mode= 'categorical')
test_data= datagen.flow_from_directory('Data/Test', color_mode= 'grayscale', target_size= (32, 32), batch_size=32, class_mode='categorical')

batch_X, batch_y= train_data.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batch_X.shape, batch_X.min(), batch_X.max()))
image_size= (32, 32, 1)
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
model.add(layers.Dense(units= 36, activation= 'softmax'))
model.summary()

#compile model
model.compile(loss='categorical_crossentropy' , metrics=['accuracy'], optimizer= 'Adam')

#train model
STEP_SIZE_TRAIN=train_data.n//train_data.batch_size
STEP_SIZE_VALID=validation_data.n//validation_data.batch_size
STEP_SIZE_TEST=test_data.n//test_data.batch_size

trained_model= model.fit_generator(generator=train_data,
                                    steps_per_epoch=STEP_SIZE_TRAIN,
                                    validation_steps= STEP_SIZE_VALID,
                                    validation_data=validation_data,
                                    epochs=5,
                                    )

#evaluate model
model.evaluate_generator(generator=validation_data, steps=STEP_SIZE_VALID)

#predict output
test_data.reset()
pred=model.predict_generator(test_data, steps=STEP_SIZE_TEST, verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_data.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
#save predictions
filenames=test_data.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)

#save model weights and architecture
save_model_to_h5 = base_path + 'trained_model' + '.hdf5'
save_model_to_json = base_path + 'trained_model' + '.json' 

#print accuracy  and loss curves
plt.figure(figsize=[10,8])
plt.plot(model_train.history['loss'], 'g', linewidth=0.5)
plt.plot(model_train.history['val_loss'], 'r', linewidth=3.0)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Training Loss', 'Validation Loss'])
plt.title('Loss Curves')

plt.figure(figsize=[10,8])
plt.plot(model_train.history['acc'], 'g', linewidth=0.5)
plt.plot(model_train.history['val_acc'], 'r', linewidth=3.0)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Training Accuracy', 'Validation Accuracy'])
plt.title('Accuracy Curves')