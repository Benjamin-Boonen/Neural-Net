#A program to load the mnist dataset and train our own neural network on

from neural import *
import numpy as np
import struct, random, math
from tqdm import tqdm

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

#images are 28x28, 60k training and 10k testing images
n = Network(shape = [784, 30, 25, 25, 20, 10], is_random = True)

print("Starting training...")

amt = 1_000_000
#Change value!!!
for i in tqdm(range(amt)):
    x = random.randint(0, len(train_images)-1)
    b_propagation(n, train_images[x], train_labels[x], learning_rate = 1)

for i in range(len(test_images) // 500):
    output = f_propagation(n, test_images[i])

    maxValue = 0
    maxIndex = 0
    for k in range(len(output)):
        if output[k] >= maxValue:
            maxValue = output[k]
            maxIndex = k
        
    print("Output:", maxIndex, "| expected value:", test_labels[i])