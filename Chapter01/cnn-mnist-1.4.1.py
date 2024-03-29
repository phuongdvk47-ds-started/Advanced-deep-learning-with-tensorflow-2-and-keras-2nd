''' CNN MNIST digits classification
3-layer CNN for MNIST digits classification
First 2 layers - Conv2D-ReLU-MaxPool
3rd layer - Conv2D-ReLU-Dropout
4th layer - Dense(10)
Output Activation - softmax
Optimizer - Adam
99.4% test accuracy in 10epochs
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, Dense, Dropout
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow.keras.datasets import mnist

# load mnist dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))

# convert to one-hot vector
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# input image dimensions
image_size = x_train.shape[1]
# resize and normalize
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
# image is processed as is (square grayscale)
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
pool_size = 2
filters = 64
dropout = 0.2

def get_model():
    # model is a stack of CNN-ReLU-MaxPooling
    model = tf.keras.models.Sequential()
    # Add Convolutional layer with filters (64) kernels, each kernel with size 3*3
    # filters is number of kernels
    model.add(tf.keras.layers.Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size))
    model.add(tf.keras.layers.Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size))
    model.add(tf.keras.layers.Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    activation='relu'))
    model.add(tf.keras.layers.Flatten())
    # dropout added as regularizer
    model.add(tf.keras.layers.Dropout(dropout))
    # output layer is 10-dim one-hot vector
    model.add(tf.keras.layers.Dense(num_labels))
    model.add(tf.keras.layers.Activation('softmax'))
    model.summary()
    # enable this if pydot can be installed
    # pip install pydot
    #plot_model(model, to_file='cnn-mnist.png', show_shapes=True)

    # loss function for one-hot vector
    # use of adam optimizer
    # accuracy is good metric for classification tasks

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

model_gpu = get_model()
model_gpu.summary()
# train the network
#model_gpu.fit(x_train, y_train, epochs=10, batch_size=batch_size)
#_, acc = model_gpu.evaluate(x_test,
#                        y_test,
#                        batch_size=batch_size,
#                        verbose=0)
#print("\nTest accuracy: %.1f%%" % (100.0 * acc))
