
import tensorflow as tf
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


def get_classlabel(class_code):
    labels = {0: 'glass', 1: 'metal', 2: 'paper', 3: 'plastic', 4: 'trash'}
    return labels[class_code]
def get_images(directory):
    Images = []
    Labels = []  # 0 for glass , 1 metal, 2 for paper, 3 for plastic, 4 for trash
    label = 0
    for image_file in os.listdir(directory ):  # Extracting the file name of the image from Class Label folder
            image = cv2.imread(directory + r'/' + image_file)  # Reading the image (OpenCV)
            image = cv2.resize(image, (150, 150))  # Resize the image, Some images are different sizes. (Resizing is very Important)
            Images.append(image)
    return shuffle(Images, random_state=817328462)  # Shuffle the dataset you just prepared.


def start(n):
    if(n==1):
        MODEL_FILENAME = "model.hdf5"
        CAPTCHA_IMAGE_FOLDER = "testtrash"


        # Load up the model labels (so we can translate model predictions to actual letters)

        lb =  {0: 'glass', 1: 'metal', 2: 'paper', 3: 'plastic', 4: 'trash'}
        # Load the trained neural network
        model = tf.keras.models.load_model(MODEL_FILENAME)

        image = cv2.imread("testtrash/test.jpg")

        image = cv2.resize(image, (150, 150))



        #contours = cv2.findContours(image)[0]

        letter_image_regions = []


            # Get the rectangle that contains the contour
        #(x, y, w, h) = cv2.boundingRect(contours[0])
        #letter_image_regions.append((x, y, w, h))

            # Create an output image and a list to hold our predicted letters
       # output = cv2.merge([image] * 3)
        predictions = []

        # Ask the neural network to make a prediction
        pred_images = get_images('C:/Users/owais/Desktop/wasteclass 2.0/testtrash/')
        pred_images = np.array(pred_images)
        pred_images.shape
        pred_image = np.array([pred_images[0]])
        pred_class = get_classlabel(model.predict_classes(pred_image)[0])
        pred_prob = model.predict(pred_image).reshape(6)
            # Convert the one-hot-encoded prediction back to a normal letter



            # draw the prediction on the output image
        #   cv2.rectangle(output, (x-2, y-2), (x + w, y ), (0, 255, 0), 1)
         #   cv2.putText(output, letter, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Print the captcha's text

        print(pred_class)
        print(pred_prob)
        return (image)
            # Show the annotated image

