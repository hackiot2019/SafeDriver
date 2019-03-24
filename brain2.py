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


def label_image(img):
    if "open" in img:
        return [1, 0]
    elif "closed" in img:
        return [0, 1]


def train_data_loder():
    training_data = []
    for img in tqdm(os.listdir(path=TRAIN_DIR)):
        img_lable = label_image(img)
        path_to_img = os.path.join(TRAIN_DIR, img)
        img = cv2.resize(cv2.imread(
            path_to_img, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        training_data.append([np.array(img), np.array(img_lable)])

    shuffle(training_data)
    np.save("training_data_new.npy", training_data)
    return training_data


def testing_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        img_labels = img.split(".")[0]
        path_to_img = os.path.join(TEST_DIR, img)
        img = cv2.resize(cv2.imread(
            path_to_img, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
        test_data.append([np.array(img), np.array(img_labels)])

    shuffle(test_data)
    np.save("test_dataone.npy", test_data)
    return test_data


train_data_loder()
testing_data()
train_data_g = np.load('training_data_new.npy')


tf.reset_default_graph()

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

if os.path.exists("{}.meta".format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print("Model Loaded")

train = train_data_g[:-1000]
test = train_data_g[-1000:]
# This is our Training data
X = np.array([i[0] for i in train]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
Y = [i[1] for i in train]

# This is our Training data
test_x = np.array([i[0] for i in test]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
test_y = [i[1] for i in test]

model.fit(X, Y, n_epoch=10, validation_set=(test_x,  test_y),
          snapshot_step=1000, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)
test_data = np.load("test_dataone.npy")
