import random
import numpy as np
import ast
import math


"""
No 'neuron' or 'node' is defined, all will be held in matrices in the Layer class.
A layer contains:
- a horizontal matrix with the values of the nodes
- a (set of) vertical matrix(es) containing the weights for the next layer (if not output)
- " with the biases for the next layer (if not output)
"""
        

class Layer:
    def __init__(self, size, next_layer_size=0, index=0, outp_=False, w=True, b=True, r=False):
        self.values = np.zeros(size).astype(int).tolist()
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
                    for i in range(next_layer_size):
                        self.biases.append((random.random() * 2)-1)
            else:
                if w:
                    self.weights = np.ones((self.size, next_layer_size))
                if b:
                    self.biases = np.zeros(next_layer_size)
        

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


    def mod_network(self, factor, is_random=False, fine_tune_mode=False):

        if fine_tune_mode:
            for i in range(len(self.layers)):
                if is_random:
                    self.layers[i].weights *= (1 + np.random.uniform(-factor, factor, self.layers[i].weights.shape))
                else:
                    self.layers[i].weights *= np.full_like(self.layers[i].weights, factor)
        else:
            for i in range(len(self.layers)):
                if is_random:
                    self.layers[i].weights += (1 + np.random.uniform(-factor, factor, self.layers[i].weights.shape))
                else:
                    self.layers[i].weights += np.full_like(self.layers[i].weights, factor)

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
    

def save_network(network: Network, filename="network.nn"):
    data = [network.shape]
    
    # print(data)
    with open(filename, 'w') as f:
        f.write(str(data) + "\n")
    with open(filename, 'a') as f:
        for i in range(len(network.shape)):
            f.write(str(network.layers[i].values.tolist()) + "\n")
            f.write(str(network.layers[i].weights.tolist()) + "\n")
            f.write(str(network.layers[i].biases.tolist()) + "\n")

def load_network(filename: str):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # print(lines)
    n = Network(shape=[])
    for i in range(len(lines)):
        if i == 0:
            shape = lines[i][1:-2]
            shape = ast.literal_eval(shape)
            n.shape = shape
            continue

        if i % 3 == 1:
            # print("value")
            value = ast.literal_eval(lines[i])
            n.layers.append(Layer(len(value), outp_=True))
            n.layers[int((i-1)/3)].value = value

        if i % 3 == 2:
            # print("weight")
            n.layers[int((i-2)/3)].weights = ast.literal_eval(lines[i])

        if i % 3 == 0:
            # print("bias")
            n.layers[int((i-3)/3)].biases = ast.literal_eval(lines[i])
    return n

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z):
    s = 1 / (1 + np.exp(-z))
    return s * (1 - s)
    
def calc_loss(recieved, expected):
    difference = np.array(recieved) - np.array(expected)
    loss = np.square(difference)
    return loss

def calc_cost(recieved, expected):
    loss = calc_loss(recieved, expected)
    return np.sum(loss)

def f_propagation(network: Network, values):
    if len(values) != len(network.layers[0].values):
        raise SyntaxError(f"Input length {len(values)} != first layer size {len(network.layers[0].values)}")
    
    network.layers[0].values = values
    
    for i in range(len(network.layers)-1):
        z = np.dot(network.layers[i].values, network.layers[i].weights) + network.layers[i].biases
        network.layers[i+1].z_values = z  # store pre-activation
        network.layers[i+1].values = sigmoid(z).tolist()
    
    return network.layers[-1].values

def b_propagation(network: Network, x, y, learning_rate=0.1):
    # Forward pass
    output = f_propagation(network, x)

    # Convert to np arrays for math
    y = np.array(y)
    a_last = np.array(output)
    z_last = np.array(network.layers[-1].z_values)

    # Compute delta at output layer
    delta = (a_last - y) * sigmoid_derivative(z_last)

    # Store deltas for each layer (same size as layer)
    deltas = [None] * len(network.layers)
    deltas[-1] = delta

    # Backpropagate deltas
    for l in range(len(network.layers)-2, 0, -1):
        z = np.array(network.layers[l].z_values)
        sp = sigmoid_derivative(z)
        delta = np.dot(deltas[l+1], np.array(network.layers[l].weights).T) * sp
        deltas[l] = delta

    # Update weights and biases
    for l in range(len(network.layers)-1):
        a = np.array(network.layers[l].values, ndmin=2)
        d = np.array(deltas[l+1], ndmin=2)

        # Gradient descent update
        network.layers[l].weights -= learning_rate * a.T.dot(d)
        network.layers[l].biases -= learning_rate * d.flatten()

# Example usage
if __name__ == "__main__":
    n = Network(shape=[2, 3, 6, 1], is_random=True)
    
    # Train on XOR
    data = [
        ([0,0],[0]),
        ([0,1],[1]),
        ([1,0],[1]),
        ([1,1],[0])
    ]
    
    for epoch in range(10000):
        x, y = random.choice(data)
        b_propagation(n, x, y, learning_rate=0.5)
    
    for x, y in data:
        print(x, round(f_propagation(n, x)[0]), "expected:", y)
