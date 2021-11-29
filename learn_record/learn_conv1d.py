from tensorflow.keras.layers import Conv1D, InputLayer, Convolution1D
import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(123)
tf.random.set_seed(123)
x = np.array([[[1, 2], [2, 1]]], dtype='float32')
y = Conv1D(2, 1, padding='same', activation='relu', data_format='channels_last')(x)
print(y.shape)
print(y)
