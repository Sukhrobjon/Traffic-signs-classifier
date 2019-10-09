# Loading Data
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, color, exposure, transform
from sklearn.model_selection import train_test_split
import os
import glob
import h5py

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras import backend as K
K.set_image_data_format('channels_first')
# %matplotlib inline

# CUSTOM
from hdf5_script import X_and_y


NUM_CLASSES = 43
IMG_SIZE = 48
# to run the functions only once


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


# create model
# model compatible today's keras
@run_once
def cnn_simple_model():
    """
    """

    INPUT_SHAPE = (3, 48, 48)
    model = Sequential()

    # 32
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', padding='same',
                     input_shape=INPUT_SHAPE))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # 64
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model


# calling X and y and call this only once
def X_and_y_train():
    if not os.path.exists('./X.h5'):
        X_train, y_train = X_and_y()
    else:
        with h5py.File('./X.h5') as hf:
            X_train = hf['imgs'][:]
            y_train = hf['labels'][:]
            # print(X_train)
            # print(y_train)
    return X_train, y_train


X_train, y_train = X_and_y_train()

print(f"X shape: {X_train.shape}")
print(f"Y shape: {y_train.shape}")

# # compile the model
# model = cnn_simple_model()

# # let's train the model using SGD + momentum.
# # source on learning rate: https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
# lr = 0.01
# sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)


# model.compile(loss='categorical_crossentropy',
#               optimizer=sgd,
#               metrics=['accuracy'])

# print("Compile the model")


# # @run_once
# def lr_schedule(epoch):
#     return lr*(0.1**int(epoch/10))


# # fit the model
# batch_size = 32
# epochs = 5
# callbacks_list = [LearningRateScheduler(lr_schedule),
#                 ModelCheckpoint('simple_model_gtsrb.h5', save_best_only=True)]

# model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_split=0.2,
#           shuffle=True,
#           callbacks=callbacks_list
# )
