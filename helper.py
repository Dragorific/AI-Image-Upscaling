import glob
import cv2
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def mean(arr):
    return sum(arr) / len(arr)

# SSIM without any library functions
def ssim_loss_python(y_true, y_pred, K1=0.01, K2=0.03, L=255.0):
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    height = len(y_true)
    width = len(y_true[0])

    mu_true = mean([mean(row) for row in y_true])
    mu_pred = mean([mean(row) for row in y_pred])

    sigma_true_sq = mean([mean([(y_true[i][j] - mu_true) ** 2 for j in range(width)]) for i in range(height)])
    sigma_pred_sq = mean([mean([(y_pred[i][j] - mu_pred) ** 2 for j in range(width)]) for i in range(height)])

    sigma_true_pred = mean([mean([(y_true[i][j] - mu_true) * (y_pred[i][j] - mu_pred) for j in range(width)]) for i in range(height)])

    ssim_n = (2 * mu_true * mu_pred + C1) * (2 * sigma_true_pred + C2)
    ssim_d = (mu_true ** 2 + mu_pred ** 2 + C1) * (sigma_true_sq + sigma_pred_sq + C2)

    ssim = ssim_n / ssim_d
    loss = 1.0 - ssim

    return loss

# SSIM loss using tensorflow and numpy function
def ssim_loss(y_true, y_pred, K1=0.01, K2=0.03, L=255.0):
    y_true = tf.cast(y_true, tf.float32)
    
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu_true = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    mu_pred = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)

    sigma_true_sq = tf.reduce_mean(tf.square(y_true - mu_true), axis=[1, 2])
    sigma_pred_sq = tf.reduce_mean(tf.square(y_pred - mu_pred), axis=[1, 2])

    sigma_true_pred = tf.reduce_mean((y_true - mu_true) * (y_pred - mu_pred), axis=[1, 2])

    ssim_n = (2 * mu_true * mu_pred + C1) * (2 * sigma_true_pred + C2)
    ssim_d = (tf.square(mu_true) + tf.square(mu_pred) + C1) * (sigma_true_sq + sigma_pred_sq + C2)

    ssim = ssim_n / ssim_d
    loss = tf.reduce_mean(1.0 - ssim, axis=0)

    return loss

# A combined loss function that combines both SSIM and MSE
def combined_loss(y_true, y_pred, alpha=0.5):

    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    ssim_loss = 1.0 - tf.image.ssim(y_true, y_pred, max_val=255)
    return alpha * mse_loss + (1 - alpha) * ssim_loss

# Get model doesn't change, we can make the same model base for each prediction model
def get_model():
    input = Input(shape=(None, None, 1))
    x = Conv2D(32, 3, activation='relu', padding='same')(input)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(1, 3, activation=None, padding='same')(x)
    x = Activation('tanh')(x)
    x = x * 127.5 + 127.5

    model = Model([input], x)
    model.summary()
    return model

def get_model_uv():
    input = Input(shape=(None, None, 1))
    x = Conv2D(32, 3, activation='relu', padding='same')(input)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = Conv2D(32, 3, activation='relu', padding='same')(x)
    x = Conv2D(1, 3, activation=None, padding='same')(x)

    model = Model([input], x)
    model.summary()
    return model

# We can create diff data for each channel and return it
def get_data():
    # Initialize our lists
    y_x = []
    y_y = []
    u_x = []
    u_y = []
    v_x = []
    v_y = []
    for img_dir in tqdm(glob.glob('./DIV2K_train_HR/*.png')):
        img = cv2.imread(img_dir)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        y_channel = img_ycrcb[:, :, 0]
        u_channel = img_ycrcb[:, :, 1]
        v_channel = img_ycrcb[:, :, 2]
        
        y_out = cv2.resize(y_channel, (128, 128), interpolation=cv2.INTER_AREA)
        y_in = cv2.resize(y_out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        u_out = cv2.resize(u_channel, (128, 128), interpolation=cv2.INTER_AREA)
        u_in = cv2.resize(u_out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


        v_out = cv2.resize(v_channel, (128, 128), interpolation=cv2.INTER_AREA)
        v_in = cv2.resize(v_out, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        
        y_x.append(y_in)
        y_y.append(y_out)

        u_x.append(u_in)
        u_y.append(u_out)
        
        v_x.append(v_in)
        v_y.append(v_out)

    y_x = np.array(y_x)
    y_y = np.array(y_y)

    u_x = np.array(u_x)
    u_y = np.array(u_y)

    v_x = np.array(v_x)
    v_y = np.array(v_y)

    return y_x, y_y, u_x, u_y, v_x, v_y

def ssim_rgb(img1, img2, max_val=255.0, K1=0.01, K2=0.03):
    img1 = tf.expand_dims(img1, axis=0)
    img2 = tf.expand_dims(img2, axis=0)

    img1 = tf.cast(img1, tf.float32)
    img2 = tf.cast(img2, tf.float32)

    C1 = (K1 * max_val) ** 2
    C2 = (K2 * max_val) ** 2

    mu_true = tf.reduce_mean(img1, axis=[1, 2], keepdims=True)
    mu_pred = tf.reduce_mean(img2, axis=[1, 2], keepdims=True)

    sigma_true_sq = tf.reduce_mean(tf.square(img1 - mu_true), axis=[1, 2])
    sigma_pred_sq = tf.reduce_mean(tf.square(img2 - mu_pred), axis=[1, 2])

    sigma_true_pred = tf.reduce_mean((img1 - mu_true) * (img2 - mu_pred), axis=[1, 2])

    ssim_n = (2 * mu_true * mu_pred + C1) * (2 * sigma_true_pred + C2)
    ssim_d = (tf.square(mu_true) + tf.square(mu_pred) + C1) * (sigma_true_sq + sigma_pred_sq + C2)

    ssim = ssim_n / ssim_d
    return tf.reduce_mean(ssim, axis=0)