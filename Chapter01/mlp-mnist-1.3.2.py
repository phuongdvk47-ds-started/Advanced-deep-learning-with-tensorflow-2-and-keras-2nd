'''
A MLP network for MNIST digits classification
98.3% test accuracy in 20 epochs
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf



output_dir = "output"
isExist = os.path.exists(output_dir)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(output_dir)
   print("The new directory '", output_dir, "' is created!")

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

''' The shape of an array is the number of elements in each dimension.
>>> x_train.shape
(60000, 28, 28)
'''

# compute the number of labels
num_labels = len(np.unique(y_train))

''' unique labels : 0 - 9 digits
>>> num_labels
10
'''

# convert to one-hot vector
''' before convert
>>> y_train[:5]
array([5, 0, 4, 1, 9], dtype=uint8)
>>> y_test[:5]
array([7, 2, 1, 0, 4], dtype=uint8)
'''
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

''' After convert
>>> y_train[:5]
array([[0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=float32)
>>> y_test[:5]
array([[0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]], dtype=float32)
'''

# image dimensions (assumed square)
image_size = x_train.shape[1]
input_size = image_size * image_size

# resize (array 3d to 2d) and normalize
'''
Reshaping means changing the shape of an array.
x_train: (60000, 28, 28) --> (60000, 784)
Normalize value of x_train and x_test
int [0..255] --> float [0..1]
'''
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255

# network parameters
batch_size = 128
hidden_units = 256
dropout = 0.45

# model is a 3 layers MLP with ReLU and dropout after each layer
#
# Dense(784) -> Activation('relu') -> Dropout(0.45) -> Dense(256) ->  Activation('relu') --> Dropout(0.45) -> Dense(10) --> Activation('softmax')
#
'''
* Dense layer is the regular deeply connected neural network layer. It is most common and frequently used layer. 
  Dense layer does the below operation on the input and return the output
  
'''

model = tf.keras.models.Sequential()
## first layer with N = 28*28 = 784
model.add(tf.keras.layers.Dense(hidden_units, activation='relu', input_dim = input_size))
model.add(tf.keras.layers.Dropout(dropout))
## second layer with N = 256
model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
model.add(tf.keras.layers.Dropout(dropout))
## third layer with N = 10 (0..9)
model.add(tf.keras.layers.Dense(num_labels, activation='softmax'))
model.summary()

#tf.keras.utils.plot_model(model, to_file= output_dir + '/mpl-mnist.png', show_shapes=True)

# Loss function for one-hot vector
# use of adam optimizer
# accuracy is good mectric for classification tasks
model.compile(loss = 'categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
# train the network
model.fit(x_train, y_train, epochs=20, batch_size=batch_size) 

# validate the model on test dataset to determine generalization
_, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

print("\n Test accuracy: %.1f%%" % (100.0 * acc))

