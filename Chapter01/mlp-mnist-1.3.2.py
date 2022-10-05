'''
A MLP network for MNIST digits classification
98.3% test accuracy in 20 epochs
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from statistics import mode

import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist


output_dir = "output"
isExist = os.path.exists(output_dir)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(output_dir)
   print("The new directory '", output_dir, "' is created!")

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

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

model = Sequential()
## first layer with N = 28*28 = 784
model.add(Dense(hidden_units, input_dim = input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
## second layer with N = 256
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
## third layer with N = 10 (0..9)
model.add(Dense(num_labels))
# this is the output for one-hot vector
model.add(Activation('softmax'))
model.summary()
plot_model(model, to_file= output_dir + '/mpl-mnist.png', show_shapes=True)


