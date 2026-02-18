#A program to load the mnist dataset and train our own neural network on

from neural import *
import numpy as np
import struct
import random
import math

#Load functions
#stole this part, no idea how it works
def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows * cols)
        images = images / 255.0  # normalize 0-1
        return images
    
def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def one_hot(label, size=10):
    vec = np.zeros(size)
    vec[label] = 1
    return vec

#Actually load the imgages and labels
#train images are for training and testing for the final test (obv)
train_images = load_images("Neural/MNIST Dataset/mnist/train-images.idx3-ubyte")
train_labels = load_labels("Neural/MNIST Dataset/mnist/train-labels.idx1-ubyte")

test_images = load_images("Neural/MNIST Dataset/mnist/t10k-images.idx3-ubyte")
test_labels = load_labels("Neural/MNIST Dataset/mnist/t10k-labels.idx1-ubyte")

n = Network(shape = [784, 5, 3, 3, 5, 10], is_random = True)

print("Starting training...")

j = 0;
for i in range(1000000):
    x = random.randint(0, len(train_images)-1)
    b_propagation(n, train_images[x], train_labels[x], learning_rate = 1)
    j += 1
    if j == 500000:
        j = 0
        print("training:", math.ceil((i/10000)), "%")



for i in range(len(test_images)):
    print(test_images[i], f_propagation(n, test_images[i]), "expected value:", test_labels[i])