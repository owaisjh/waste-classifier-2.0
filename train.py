import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
import os.path
import pickle

def get_images(directory):
    Images = []
    Labels = []  # 0 for glass , 1 metal, 2 for paper, 3 for plastic, 4 for trash
    label = 0

    for labels in os.listdir(directory):  # Main Directory where each class label is present as folder name.
        if labels == 'glass':  # Folder contain Glacier Images get the '2' class label.
            label = 0
        elif labels == 'metal':
            label = 1
        elif labels == 'paper':
            label = 2
        elif labels == 'plastic':
            label = 3
        elif labels == 'trash':
            label = 4


        for image_file in os.listdir(directory + labels):  # Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory + labels + r'/' + image_file)  # Reading the image (OpenCV)
            image = cv2.resize(image, (150, 150))  # Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
            Labels.append(label)

    return shuffle(Images, Labels, random_state=817328462)  # Shuffle the dataset you just prepared.


Images, Labels = get_images('data/') #Extract the training images from the folders.

Images = np.array(Images) #converting the list of images to numpy array.
Labels = np.array(Labels)

print("Shape of Images:",Images.shape)
print("Shape of Labels:",Labels.shape)

model = Models.Sequential()

model.add(Layers.Conv2D(200,kernel_size=(3,3),activation='relu',input_shape=(150,150,3)))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Conv2D(180,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(140,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(100,kernel_size=(3,3),activation='relu'))
model.add(Layers.Conv2D(50,kernel_size=(3,3),activation='relu'))
model.add(Layers.MaxPool2D(5,5))
model.add(Layers.Flatten())
model.add(Layers.Dense(180,activation='relu'))
model.add(Layers.Dense(100,activation='relu'))
model.add(Layers.Dense(50,activation='relu'))
model.add(Layers.Dropout(rate=0.5))
model.add(Layers.Dense(6,activation='softmax'))

model.compile(optimizer=Optimizer.Adam(lr=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.summary()
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
#Utils.plot_model(model,to_file='model.png',show_shapes=True)

#trained = model.fit(Images,Labels,epochs=35,validation_split=0.30)

#model.save("model.hdf5")