import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def plot_loss(hist,plt):
    loss = hist['loss']
    val_loss = hist['val_loss']
    loss = np.sqrt(np.array(loss)) * 48
    val_loss = np.sqrt(np.array(val_loss)) * 48
    plt.plot(loss,"--", linewidth=3, label="train:"+name)
    plt.plot(val_loss, linewidth=3, label="val:"+name)
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("log loss")
    plt.show()


def plot_sample(x, y):
    fig = plt.figure()
    img = x.reshape(96, 96)
    plt.imshow(img, cmap='gray')
    plt.scatter(y[0::2], y[1::2], marker='o', s=25, c='g')
    plt.axis('off')
    plt.show()


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


def load_data():
    X = np.load('../data/X.npy')
    y = np.load('../data/y.npy')
    return X, y


if __name__ == '__main__':

    # --------------- Plotting -----------------
    i = 5
    X, y_hat = load_data()
    print('X.shape: ', X.shape)
    print('y_hat.shape: ', y_hat.shape)
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
    print(ix_good)
    print(ix_bad)
    print(errors)
    print('predictions completed')
    plot_sample(X[ix_bad], y[ix_bad])
    plot_sample(X[ix_good], y[ix_good])
    
    

    # --------------- Training/Evaluating -----------------
    '''
    retrain_model = False
    preprocess_data = False
    epochs = 50
    batch_size = 256
    valid_size = 0.2
    test_size = 0.3


    X = None
    y = None
    if not preprocess_data: X, y = load_data()
    else: X, y = preprocess()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    print('Train shape: ', X_train.shape)
    print('Test shape: ', X_test.shape)

    model = None
    if retrain_model:
        model = createModel()
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        hist = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_split = valid_size)
        model.save_weights("model.h5")
        print("Saved model to disk")
    else:
        model.load_weights("model.h5")
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        print("Loaded model from disk")


    predictions = model.predict(X_test)
    print('First prediction:', predictions[0])

    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test MAE', score[1])
    '''


