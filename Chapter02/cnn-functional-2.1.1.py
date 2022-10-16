'''
  Using API Function to build CNN
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

## load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# from sparse label to categorical
num_labels = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

#reshare and normalize input images
# x_train.shape = (count, n, n)
image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
#normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 64
dropout = 0.3

# use functional API to build cnn layers
inputs = tf.keras.layers.Input(shape=input_shape)
y = tf.keras.layers.Conv2D(filters = filters,
                      kernel_size = kernel_size,
                      activation = 'relu')(inputs)
y = tf.keras.layers.MaxPooling2D()(y)
# Hidden layers 1
y = tf.keras.layers.Conv2D(filters = filters,
                      kernel_size = kernel_size,
                      activation = 'relu')(y)
y = tf.keras.layers.MaxPooling2D()(y)

y = tf.keras.layers.Conv2D(filters = filters,
                      kernel_size = kernel_size,
                      activation = 'relu')(y)
# image to vector before connecting to dense layer
y = tf.keras.layers.Flatten()(y)
# dropout regularization
y = tf.keras.layers.Dropout(dropout)(y)

outputs = tf.keras.layers.Dense(num_labels, activation='softmax')(y)

# build the model by supplying inputs/outputs
model = tf.keras.models.Model(inputs = inputs, outputs = outputs)
# network model in text
model.summary()

# classifier loss, Adam optimizer, classifier accuracy
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

# Training
model.fit(x_train, y_train, validation_data =(x_test, y_test), epochs = 20, batch_size= batch_size)

# model accuracy on test dataset
score = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)
print("\nTest accuracy: %.1f%%", (100.0 * score[1]))
