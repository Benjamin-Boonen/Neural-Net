import numpy as np
import struct

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

train_images = load_images("Neural/MNIST Dataset/mnist/train-images.idx3-ubyte")
train_labels = load_labels("Neural/MNIST Dataset/mnist/train-labels.idx1-ubyte")

test_images = load_images("Neural/MNIST Dataset/mnist/t10k-images.idx3-ubyte")
test_labels = load_labels("Neural/MNIST Dataset/mnist/t10k-labels.idx1-ubyte")

print(train_labels[0])