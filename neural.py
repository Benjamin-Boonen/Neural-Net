import math
import random
import numpy as np


"""
No 'neuron' or 'node' is defined, all will be held in matrices in the Layer class.
A layer contains:
- a horizontal matrix with the values of the nodes
- a (set of) vertical matrix(es) containing the weights for the next layer (if not output)
- " with the biases for the next layer (if not output)
"""
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
    
class Layer:
    def __init__(self, size, next_layer_size, index, outp_=False, activ_fn=None, w=True, b=True, r=False):
        self.values = np.zeros(size)
        self.weights = np.empty(0)
        self.biases = np.empty(0)
        self.size = size

        if r:
            if w:
                for i in range(self.size):
                    np.insert(self.weights, np.empty(0))
                    for j in range(next_layer_size):
                        np.insert(self.weights[i], (random.random * 2)-1)
            if b:
                for i in range(self.size):
                    self.biases[i] = np.empty(0)
                    for j in range(next_layer_size):
                        self.biases[i][j] = (random.random * 2)-1
        else:
            if w:
                self.weights = np.ones((next_layer_size, self.size))
            if b:
                self.biases = np.zeros((next_layer_size, self.size))
        

class Network:
    pass


# Example usage
if __name__ == "__main__":
    l = Layer(3, 5, 1, r=True)
    print(l.weights)
    print(l.biases)