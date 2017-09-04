#%%
import numpy as np
from time import time

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
#%%
class NeuralNet():
    """
    Neural Network
    """
    def __init__(self, num_inputs):
        np.random.seed(1)
        self.num_inputs = num_inputs
        self.weights = []

    def __repr__(self):
        repr_str = ""
        for layer in self.weights:
            repr_str += str(layer) + "\n"
        return repr_str
    
    def __makeLayer(self, size):
        return (2 * np.random.random((size, 1)) - 1)

    def addLayer(self, layer_size):
        if len(self.weights) == 0:
            self.weights.append(self.__makeLayer(layer_size*self.num_inputs))
        else:
            self.weights.append(self.__makeLayer(layer_size*self.weights[-1].shape[0])) 
    
    def process(self, inputs, mode=True):
        if mode:
            return relu(np.dot(inputs, self.weights))
        return sigmoid(np.dot(inputs, self.weights))
    
    def propagate(self, inputs):
        for layer in self.weights:
            print(layer.shape[0])