import cv2
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from helper import *

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
img = cv2.imread('./DIV2K_valid_HR/0816.png')
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
upscaled_rgb = cv2.cvtColor(upscaled_ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# Convert the img_compare variable to RGB
img_compare_rgb = cv2.cvtColor(img_compare, cv2.COLOR_BGR2RGB)

# Create the image to make SSIM comparison
img_ssim = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
img_ssim_rgb = cv2.cvtColor(img_ssim, cv2.COLOR_BGR2RGB)

# Calculate SSIM
ssim_val = ssim_rgb(upscaled_rgb, img_ssim_rgb).numpy()
print("SSIM between upscaled_rgb and img_compare_rgb:", ssim_val)

# Display the original image and the upscaled image
plt.figure()
plt.subplot(121)
plt.imshow(img_compare_rgb)
plt.title("Original Image")
plt.subplot(122)
plt.imshow(upscaled_rgb)
plt.title("Upscaled Image")
plt.show()
