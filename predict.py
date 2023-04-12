import cv2
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# inference part! for the new image!
model = load_model('model/model2.h5')
img = cv2.imread('./DIV2K_valid_HR/low_res_image.jpg')
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y_channel = img_ycrcb[:, :, 0]

y_in = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

y = cv2.resize(y_channel, (64, 64), interpolation=cv2.INTER_AREA)

y = np.expand_dims(y, axis=0)

# if you have preprocessing you may want to apply those here!
y_upsampled = model.predict(y)

plt.subplot(211)
plt.imshow(y[0], cmap='gray')
plt.title("Original Image")
plt.subplot(212)
plt.imshow(y_upsampled[0], cmap='gray')
plt.title("Upsampled Image")
plt.show()