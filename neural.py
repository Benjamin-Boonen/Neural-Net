import random
import numpy as np
import ast


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

    def f_propagation(self, values):
        if self.layers == []:
            raise SyntaxError("The network has no associated layers, define those first.")

        if len(values) != len(self.layers[0].values):
            raise SyntaxError(f"The length of the input list needs to be the same as the first layer. Which is {len(self.layers[0].values)}")
        
        for i in range(len(self.layers)):
            if i < len(self.layers)-1:
                vector = np.dot(self.layers[i].values, self.layers[i].weights)
                vector += self.layers[i].biases
                # print("VECTOR", vector)
                self.layers[i+1].values = vector.astype(float).tolist()
            if i == len(self.layers)-1:
                return self.layers[i].values

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
            f.write(str(network.layers[i].values) + "\n")
            f.write(str(network.layers[i].weights) + "\n")
            f.write(str(network.layers[i].biases) + "\n")

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


# Example usage
if __name__ == "__main__":
    # l = Layer(3, 5, 1, r=True)
    # print(np.asmatrix(l.weights))
    # print(np.asmatrix(l.biases))
    n = Network(shape=[3, 2, 4], is_random=True)
    # n.show_network()
    print(n.f_propagation([0, 1, 2]))
    input()
    save_network(n)
    print(load_network('network.nn'))