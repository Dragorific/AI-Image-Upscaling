import glob
import cv2
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import *
from keras.models import *
from keras.layers import Dropout
from keras.optimizers import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def get_downscaling_model():
    input = Input(shape=(None, None, 1))
    x = Conv2D(128, 3, activation='relu', padding='same')(input)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(16, 3, activation='relu', padding='same')(x)
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Conv2D(1, 3, activation=None, padding='same')(x)
    x = Activation('tanh')(x)
    x = x * 127.5 + 127.5

    model = Model([input], x)
    model.summary()
    return model

def get_downscaling_data():
    x_data = []
    y_data = []

    for img_dir in tqdm(glob.glob('./DIV2K_train_HR/*.png')):
        img = cv2.imread(img_dir)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y_channel = img_ycrcb[:, :, 0]

        x_in = cv2.resize(y_channel, (512, 512), interpolation=cv2.INTER_LINEAR)
        y_out = cv2.resize(x_in, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_LINEAR)

        x_data.append(x_in)
        y_data.append(y_out)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

model_downscaling = get_downscaling_model()
x_data, y_data = get_downscaling_data()

X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)
model_downscaling.compile(loss='mse', optimizer=optimizer)

save_model_callback = tf.keras.callbacks.ModelCheckpoint('model/model_downscaling.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

batch_size = 4
epochs = 30

model_downscaling.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[save_model_callback, tbCallBack])
