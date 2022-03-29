"""
@author: Christian Pfister
https://cpfister.com
https://github.com/christianpfister43/Reading-a-Live-Ticker
used dataset: https://github.com/christianpfister43/CPD-Dataset
"""


import numpy as np
import cv2
import time
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import SGD


#%% set your custom paths and parameters here!
"""
Download the CPD dataset from https://github.com/christianpfister43/CPD-Dataset
and unpack the raw images into the im_path,
have the 'labels.csv' in the main folder
"""
im_path = './data/train'    # path to training images

#%% Prepare the training data
num_classes = 10

labels_df = pd.read_csv('labels.csv')
my_images = []
for o in range(len(labels_df)):
    label = labels_df.iloc[o]['label']
    im_name = labels_df.iloc[o]['image_name']
    im = cv2.imread(f'{im_path}/{im_name}')
    im_gray  = (255-cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))/255
    my_images.append(im_gray)
my_labels = labels_df['label']

X_train, X_test, y_train, y_test = train_test_split(my_images, my_labels, test_size=0.01,shuffle=True)

y = tf.keras.utils.to_categorical(y_train, num_classes)
x = np.reshape(X_train, (len(X_train),28,28,1))

#%% train a simple CNN
# This model has proven to be very reliable in predicting screen-parsed numbers

batch_size = 128
epochs = 25
opt = SGD(lr=0.01, momentum=0.9)

model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu',padding="same", input_shape=(28, 28, 1)))
model.add(MaxPool2D((3, 3)))
model.add(Dropout(0.4))

model.add(Conv2D(128, (2, 2),padding="same", activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=0.3)

#%% save the model

model.save('models/screen_parsing_model_cnn.h5')

