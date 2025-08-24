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
    def __init__(self, size, next_layer_size=0, index=0, outp_=False, activ_fn=None, w=True, b=True, r=False):
        self.values = np.zeros(size)
        self.weights = []
        self.biases = []
        self.size = size
        self.index = index

        if not(outp_):
            if r:
                if w:
                    for i in range(self.size):
                        self.weights.append([])
                        for j in range(next_layer_size):
                            self.weights[i].append((random.random() * 2)-1)
                if b:
                    for i in range(self.size):
                        self.biases.append([])
                        for j in range(next_layer_size):
                            self.biases[i].append((random.random() * 2)-1)
            else:
                if w:
                    self.weights = np.ones((next_layer_size, self.size))
                if b:
                    self.biases = np.zeros((next_layer_size, self.size))
        

class Network:
    def __init__(self, shape=None, weighted=True, is_random=False):

        self.layers = []

        if shape==None:
            if __name__ == "__main__":
                shape = []
                length = int(input("Amount of layers: "))

                for i in range(length):
                    if i == 0:
                        shape.append(int(input("Amount of input nodes: ")))
                        continue
                    if i == length-1:
                        shape.append(int(input("Amount of output nodes: ")))
                    else:
                        shape.append(int(input(f"Size of hidden layer {i}: ")))
            else:
                raise ValueError("No Shape was given for Network generation && Script was not __main__")

        self.shape = shape
        self.gen_network(self.shape, is_random=is_random, weighted=weighted)

    def gen_network(self, shape, is_random=False, weighted=True):
        for i in range(len(shape)):
            if i < len(shape)-1:
                l = Layer(shape[i], shape[i+1], i, r=is_random, w=weighted, b=weighted)
                self.layers.append(l)
            else:
                l = Layer(shape[i], outp_=True, index=i)
                self.layers.append(l)

    def show_network(self):
        for i in range(len(self.layers)):
            print(f"Layer {i} values:\n {self.layers[i].values}")
            print(f"Layer {i} weights:\n {self.layers[i].weights}")
            print(f"Layer {i} biases:\n {self.layers[i].biases}")






# Example usage
if __name__ == "__main__":
    # l = Layer(3, 5, 1, r=True)
    # print(np.asmatrix(l.weights))
    # print(np.asmatrix(l.biases))
    n = Network()
    n.show_network()