import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D


def createModel():
    model = Sequential()

    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    # model.add(BatchNormalization())
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30))

    return model


def plot_sample(x, y):
    fig = plt.figure()
    img = x.reshape(96, 96)
    plt.imshow(img, cmap='gray')
    plt.scatter(y[0::2], y[1::2], marker='o', s=25, c='g')
    plt.axis('off')
    plt.show()


def preprocess():
    df_data = pd.read_csv('../data/data.csv')
    df_data.fillna(method = 'ffill', inplace = True)
    imag = []
    for i in range(0,7049):
        img = df_data['Image'][i].split(' ')
        img = ['0' if x == '' else x for x in img]
        imag.append(img)
    
    image_list = np.array(imag, dtype = 'float')
    X = image_list.reshape(-1,96,96,1)
    df_data = df_data.drop('Image', axis = 1)

    y = []
    for i in range(0,7049):
        y_tmp = df_data.iloc[i,:]
        y.append(y_tmp)
    y = np.array(y, dtype = 'float')
    
    np.save('../data/X.npy', X)
    np.save('../data/y.npy', y)
    
    return X, y


if __name__ == '__main__':

    # --------------- Plotting -----------------
    X, y_hat = preprocess()
    X = X[:100]
    y_hat = y_hat[:100]
    print('X.shape: ', X.shape)
    print('y_hat.shape: ', y_hat.shape)
    print('data loaded')
    model = createModel()
    model.load_weights("model.h5")
    print('weights loaded')
    y = model.predict(X)
    print('y.shape: ', y.shape)
    print('y_hat.shape: ', y_hat.shape)
    errors = np.square(np.sum(y - y_hat, axis=1))
    print('errors.shape: ', errors.shape)
    ix_bad = np.argmax(errors)
    ix_good = np.argmin(errors)
    print('predictions completed')
    plot_sample(X[ix_bad], y[ix_bad])
    plot_sample(X[ix_good], y[ix_good])