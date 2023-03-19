from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# reshape input data (train data)
img_size = x_train.shape[1]
X_train = np.reshape(x_train, [-1, img_size, img_size, 1])
X_test =  np.reshape(x_test, [-1, img_size, img_size, 1])
# normalize input data
X_train = x_train.astype('float32') / 255
X_test = x_test.astype('float32') / 255

# categorial train & test data
Y_train = tf.keras.utils.to_categorical(y_train)
Y_test = tf.keras.utils.to_categorical(y_test)
# number of output node
number_labels = len(np.unique(y_train))

print('Dữ liệu ban đầu ', y_train[0:10])
print('Dữ liệu sau one-hot encoding ',Y_train[0:10])
print('Số lượng các nhãn ', number_labels)


# Defined model
model = tf.keras.models.Sequential()

# layer Conv2D with:
# number of kernels = 32
# kernel size = (3,3)
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='sigmoid', input_shape=(img_size, img_size, 1)))

# add more layer Conv2D
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='sigmoid'))
# downsampling with maxpooling
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

# convert tensor data to vector
model.add(tf.keras.layers.Flatten())

# add full connected layers with 128 nodes
model.add(tf.keras.layers.Dense(128, activation='sigmoid'))

# output layers
model.add(tf.keras.layers.Dense(number_labels, activation='softmax'))

model.summary()

## compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

# training model
H = model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=1)

fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
#plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['accuracy'], label='accuracy')
#plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

# 9. evaluation data
score = model.evaluate(X_test, Y_test, verbose=0)
print(score)

# 10. predict images
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
y_predict = model.predict(x_test[0].reshape(1,28,28,1))
print('Gi¡ trị dự đo¡n: ', np.argmax(y_predict))
