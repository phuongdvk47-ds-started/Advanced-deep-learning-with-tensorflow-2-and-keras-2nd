'''
Demonstrates how to sample and plot MNIST digits
using Keras API
https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras
'''

import os
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

output_dir = "output"
isExist = os.path.exists(output_dir)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(output_dir)
   print("The new directory '", output_dir, "' is created!")

'''
MNIST is a collection of handwritten digits ranging from 0 to 9. It has a training
set of 60,000 images, and 10,000 test images that are classified into corresponding
categories or labels. In some literature, the term target or ground truth is also used
to refer to the label.
In the preceding figure, sample images of the MNIST digits, each being sized at 28
x 28 - pixel, in grayscale, can be seen. To use the MNIST dataset in Keras, an API
is provided to download and extract images and labels automatically. Listing 1.3.1
demonstrates how to load the MNIST dataset in just one line, allowing us to both
count the train and test labels and then plot 25 random digit images
'''
# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# count the number of unique train labels
unique, counts = np.unique(y_train, return_counts=True)
print("Train unique: ", dict(zip(unique, counts)))
''' Output: unique labels is 0-9 digit, each digit has counts sample
    Train unique:  {0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949}
'''

# count the number of unique test labels
unique, counts = np.unique(y_test, return_counts=True)
print("Test unique: ", dict(zip(unique, counts)))
''' Output: unique labels is 0-9 digit, each digit has counts sample
    Test unique:  {0: 980, 1: 1135, 2: 1032, 3: 1010, 4: 982, 5: 892, 6: 958, 7: 1028, 8: 974, 9: 1009}
'''

# sample 25 mnist digits from train dataset
indexes = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

# plot the 25 mnist digits
plt.figure(figsize=(5,5))
for i in range(len(indexes)):
  plt.subplot(5,5, i+1)
  img = images[i]
  plt.imshow(img, cmap='gray')
  plt.axis('off')

# save to file
plt.savefig(output_dir + "/mnist-samples.png")
# show
plt.show()
plt.close('all')


