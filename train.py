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
from helper import *


# Define the parameters for each model
model_y = get_model()
model_u = get_model_uv()
model_v = get_model_uv()
y_x, y_y, u_x, u_y, v_x, v_y = get_data()

# ------------------------ Training the model for the Y channel ------------------------ #
# X_train, X_val, y_train, y_val = train_test_split(y_x, y_y, test_size=0.2, random_state=42)

# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)
# model_y.compile(loss=ssim_loss, optimizer=optimizer)

# save_model_callback = tf.keras.callbacks.ModelCheckpoint('model/model_y.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')

# tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# batch_size = 4
# epochs = 10

# model_y.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[save_model_callback, tbCallBack])

# ------------------------ Training the model for the U channel ------------------------ #
X_train, X_val, y_train, y_val = train_test_split(u_x, u_y, test_size=0.2, random_state=42)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)
model_u.compile(loss=combined_loss, optimizer=optimizer)

save_model_callback = tf.keras.callbacks.ModelCheckpoint('model/model_u.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

batch_size = 4
epochs = 10

model_u.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[save_model_callback, tbCallBack])

# ------------------------ Training the model for the V channel ------------------------ #
X_train, X_val, y_train, y_val = train_test_split(v_x, v_y, test_size=0.2, random_state=42)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-04)
model_v.compile(loss=combined_loss, optimizer=optimizer)

save_model_callback = tf.keras.callbacks.ModelCheckpoint('model/model_v.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch')

tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

batch_size = 4
epochs = 10

model_v.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[save_model_callback, tbCallBack])
