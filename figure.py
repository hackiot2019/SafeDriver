from tflearn.layers.core import input_data, dropout, fully_connected
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import os
import cv2
from random import shuffle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
TEST_DIR = 'test_data'
TRAIN_DIR = 'training_data'
LEARNING_RATE = 1e-3
MODEL_NAME = "eye-{}-{}.model".format(LEARNING_RATE, "6conv-fire")
IMAGE_SIZE = 50
convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')
# Conv Layer 1
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
# Conv Layer 2
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
# Conv Layer 3
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
# Dropout overfitting

convnet = dropout(convnet, 0.8)
# Fully Connected Layer with SoftMax as Activation Function
convnet = fully_connected(convnet, 2, activation='softmax')
# Regression for ConvNet with ADAM optimizer
convnet = regression(convnet, optimizer='adam', learning_rate=LEARNING_RATE,
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

model.load("eye-0.001-6conv-fire.model")
test_data = np.load("test_dataone.npy")
figs = plt.figure()
count = 0
total = 0
for num, data in enumerate(test_data):
    total += 1
    test_img = data[0]
    test_lable = data[1]
    test_img_feed = test_img.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    model_pred = model.predict([test_img_feed])[0]
    if np.argmax(model_pred) == 0:
        val = "Close"
        if "closed" in test_lable:
            count += 1
    else:
        val = "Open"
        if "open" in test_lable:
            count += 1
print(count/total)
