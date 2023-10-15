# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:55:39 2023

@author: Lenovo
"""

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('E:\DL\pneumonia_detection_model_final.h5')  # Replace with the path to your saved model

# Load and preprocess an individual image for testing
def load_and_preprocess_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Path to the image you want to test
image_path = r"E:\DL\archive (5)\chest_xray\chest_xray\test\NORMAL\IM-0029-0001.jpeg"

# Load and preprocess the test image
test_image = load_and_preprocess_image(image_path, target_size=(150, 150))  # Adjust the target size as needed

# Make a prediction
prediction = model.predict(test_image)

# Interpret the prediction
if prediction[0] > 0.5:
    print("The image is classified as pneumonia.")
else:
    print("The image is classified as normal.")
