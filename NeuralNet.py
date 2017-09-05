#%%
from time import time
import numpy as np

def timeit(func):
    def wrapper(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print('elapsed', after - before, 'ms')
        return rv
    return wrapper

def sigmoid(inputs, derivative=False):
    if derivative:
        return inputs * (1 - inputs)
    return 1 / (1 + np.exp(-inputs))

def relu(inputs, derivative=False):
    for value in inputs:
        if value > 0:
            if derivative:
                value = 1
            else:
                pass
        else:
            value = 0
    return inputs

class NeuralNet():
    """
    Neural Network
    """
    def __init__(self, num_inputs):
        np.random.seed(1)
        self.num_inputs = num_inputs
        self.weights = []
        self.layers = {0: num_inputs}
        self.num_layers = 0

    def __repr__(self):
        repr_str = ""
        for layer in self.weights:
            repr_str += str(layer) + "\n"
        return repr_str
    
    def __makeLayer(self, size):
        return (2 * np.random.random((size, 1)) - 1)

    def addLayer(self, layer_size):
        if self.num_layers == 0:
            self.weights.append(self.__makeLayer(layer_size*self.num_inputs))
        else:
            self.weights.append(self.__makeLayer(layer_size*self.layers[self.num_layers])) 
        self.layers[self.num_layers+1] = layer_size
        self.num_layers += 1
        

    def process(self, inputs, layer, mode=True):
        if mode:
            return relu(np.dot(inputs, layer))
        return sigmoid(np.dot(inputs, layer))
    
    def propagate(self, inputs):
        initial_inputs = inputs
        for layer in self.weights:
            initial_inputs = self.process(initial_inputs, layer).T
        return initial_inputs