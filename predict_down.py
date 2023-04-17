import cv2
import numpy as np
import os

import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import *
from keras.models import *
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from helper import *


model_downsample = load_model('model/model_downscaling.h5')

# Read the test image, create the comparison image, and convert to YUV format
img = cv2.imread('./DIV2K_valid_HR/0830.png')
img_compare = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Extract out the channels from the YUV to upscale separately
y_channel = img_ycrcb[:, :, 0]
u_channel = cv2.resize(img_ycrcb[:, :, 1], (64, 64), interpolation=cv2.INTER_AREA)
v_channel = cv2.resize(img_ycrcb[:, :, 2], (64, 64), interpolation=cv2.INTER_AREA)

y2 = cv2.resize(y_channel, (512, 512), interpolation=cv2.INTER_AREA)
y2 = np.expand_dims(y2, axis=0)

y_downsampled = model_downsample.predict(y2)

# Plot the upsampled channels with the original channels for visual comparison
plt.figure()

plt.subplot(131)                        # Y channel comparison
plt.imshow(y_downsampled[0])
plt.title("Y Channel (Downsampled w/ AI)")

plt.subplot(132)                        # U channel comparison
plt.imshow(u_channel)
plt.title("U Channel (Downsampled w/ cv2)")

plt.subplot(133)                        # V channel comparison
plt.imshow(v_channel)
plt.title("V Channel (Downsampled w/ cv2)")

plt.show()

# Combine the channels together to form the downscaled RGB image
img_combine = np.stack((y_downsampled[0].squeeze(), u_channel.squeeze(), v_channel.squeeze()), axis=-1)
img_combine = cv2.cvtColor(img_combine.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# Prepare the image downscaled with inter_area for comparison
img_compare = cv2.cvtColor(img_compare, cv2.COLOR_BGR2RGB)

# Display the original image and the upscaled image
plt.figure()
plt.subplot(121)
plt.imshow(img_compare)
plt.title("Downscaled Image (cv2.resize w/ INTER_AREA)")
plt.subplot(122)
plt.imshow(img_combine)
plt.title("Downscaled Image (Machine Learning)")
plt.show()