import cv2
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

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

# Main Driver Code
custom_objects = {
    'ssim_loss': ssim_loss,
    'combined_loss': combined_loss
}

# inference part! for the new image!
# Load the models for each channel
model_y = load_model('old_model/model_y.h5', custom_objects=custom_objects)
model_u = load_model('old_model/model_u.h5')
model_v = load_model('old_model/model_v.h5')

# Read the test image, create the comparison image, and convert to YUV format
img = cv2.imread('./DIV2K_valid_HR/0806.png')
img_compare = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Extract out the channels from the YUV to upscale separately
y_channel = img_ycrcb[:, :, 0]
u_channel = img_ycrcb[:, :, 1]
v_channel = img_ycrcb[:, :, 2]

# Resize and format for the model to predict (model is trained for 64x64 -> 128x128)
y = cv2.resize(y_channel, (64, 64), interpolation=cv2.INTER_AREA)
y = np.expand_dims(y, axis=0)

u = cv2.resize(u_channel, (64, 64), interpolation=cv2.INTER_AREA)
u = np.expand_dims(u, axis=0)

v = cv2.resize(v_channel, (64, 64), interpolation=cv2.INTER_AREA)
v = np.expand_dims(v, axis=0)

# Predict the upsampled image using the respective model
y_upsampled = model_y.predict(y)
u_upsampled = model_u.predict(u)
v_upsampled = model_v.predict(v)

# Plot the upsampled channels with the original channels for visual comparison
plt.figure()

plt.subplot(231)                        # Y channel comparison
plt.imshow(y[0], cmap='gray')
plt.title("Original Image - Y")
plt.subplot(234)
plt.imshow(y_upsampled[0], cmap='gray')
plt.title("Upsampled Image - Y")

plt.subplot(232)                        # U channel comparison
plt.imshow(u[0], cmap='gray')
plt.title("Original Image - U")
plt.subplot(235)
plt.imshow(u_upsampled[0], cmap='gray')
plt.title("Upsampled Image - U")

plt.subplot(233)                        # V channel comparison
plt.imshow(v[0], cmap='gray')
plt.title("Original Image - V")
plt.subplot(236)
plt.imshow(v_upsampled[0], cmap='gray')
plt.title("Upsampled Image - V")

plt.show()

# Combine the Y, U, and V channels of the upscaled images
upscaled_ycrcb = np.stack((y_upsampled[0].squeeze(), u_upsampled[0].squeeze(), v_upsampled[0].squeeze()), axis=-1)

# Convert the combined image from YCrCb to RGB color space
upscaled_rgb = cv2.cvtColor(upscaled_ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# Display the original image and the upscaled image
plt.figure()
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_compare, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.subplot(122)
plt.imshow(upscaled_rgb)
plt.title("Upscaled Image")
plt.show()
