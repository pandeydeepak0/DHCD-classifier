#build 2: registered error in line 74  
#(TypeError: 'str' object does not support item assignment)
#IndexError: index 1700 is out of bounds for axis 0 with size 1700


import sys, os
import cv2
from PIL import Image
import numpy as np
from IPython.display import display, Image
import matplotlib.image as imread
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img


folder = "Train/character_1_ka"
folders= 'Train/'

#to visit all the directories in Train set
onlydirs = [f for f in os.listdir(folders) if os.path.isdir(os.path.join(folders, f))]
print(len(onlydirs))
print(type(onlydirs))

print(folders.onlydirs[10])
for i in range(0 , 45):
        print(onlydirs[0])
#to visit all images in each sub-directory of Train set
onlyfiles = [f for f in os.walk(folders.onlydirs[0]) if os.path.isfile(os.path.join(folders.onydirs[0], f))]

print(len(onlyfiles))
print(type(onlyfiles))


print('Found {0} directories'.format(len(onlydirs)))
print("Found {0} images in each directory".format(len(onlyfiles)))


image_width = 32
image_height = 32

channels = 1
no_classes = 1

#convert the number of image into a numpy array
#type : numpy: <numpy.ndarray>
#len : 1700
dataset = np.ndarray(shape=(len(onlyfiles), channels, image_height, image_width),
                     dtype=np.float32)

#convert the number of folders into numpy array
directories = np.ndarray(shape=(len(onlydirs), ), dtype= np.float32)

print(type(directories))
print(len(directories))
i = 0
j = 0

for _dir in os.walk(folders):
        for _file in onlyfiles:
                img = load_img(folder + "/" + _file, color_mode='grayscale')  # this is a PIL image
                img.thumbnail((image_width, image_height))
                # Convert to Numpy Array
                x = img_to_array(img)  
                x = x.reshape((1, 32, 32))
                # Normalize
                x = (x - 32.0) / 32.0
                dataset[i] = x
                i += 1
                if i % 250 == 0:
                        print('%d images to array ' % i)

        directories[j]= 1
        j += 1
        if j % 1 ==1:
                print('%d Folders scanned' %j)
print('All Folders scanned')
