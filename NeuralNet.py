#%%
from time import time
import numpy as np

LEAKY_RELU = True

def timeit(func):
    def wrapper(*args, **kwargs):
        before = time()
        rv = func(*args, **kwargs)
        after = time()
        print('elapsed', after - before, 'ms')
        return rv
    return wrapper

def leaky_relu(input):
    if input > 0:
        return input
    else:
        return input*0.01

def relu_derivative(input):
    if input > 0:
        return 1
    else:
        if LEAKY_RELU:
            return 0.01
        else:
            return 0

def relu(inputs, derivative=False):
    #print(inputs)
    if LEAKY_RELU:
        if derivative:
            return np.apply_along_axis(relu_derivative, 0, inputs)
        else:
            return np.apply_along_axis(leaky_relu, 0, inputs)
    else:
        if derivative:
            return np.apply_along_axis(relu_derivative, 0, inputs)
        else:
            return np.maximum(inputs, 0, inputs)

class NeuralNet():
    """
    Neural Network
    """
    def __init__(self, num_inputs):
        np.random.seed(4)
        self.num_inputs = num_inputs
        self.weights = []
        self.layers = {0: num_inputs}
        self.values = {}
        self.num_layers = 0

    def __repr__(self):
        repr_str = ""
        for key in self.values:
            repr_str += str(self.values[key]) + "\n"
        return repr_str

    def get_sum_squared_error(self, outputs, targets):
        return np.sum(np.power(outputs - targets, 2))

    def get_error(self, targets, results, squared=False):
        if squared:
            return np.power(results - targets, 2)
        else:
            return results - targets

    def __make_layer(self, size):
        return (2 * np.random.random((size, 1)) - 1)

    def add_layer(self, layer_size):
        if self.num_layers == 0:
            self.weights.append(self.__make_layer(layer_size*self.num_inputs))
        else:
            self.weights.append(self.__make_layer(layer_size*self.layers[self.num_layers])) 
        self.layers[self.num_layers+1] = layer_size
        self.num_layers += 1
        
    def process(self, inputs, layer):
        return relu(np.dot(inputs, layer))

    def back_propagate(self, outputs, targets):
        deltas = {}
        # Delta of output Layer
        deltas[self.num_layers-1] = self.get_error(outputs[self.num_layers-1], targets)
        for layer_id in reversed(range(self.num_layers-2)):
            print(deltas)

    @timeit
    def train(self, inputs, targets, iterations):
        error = []
        for _ in range(iterations):
            for i in range(len(inputs)):
                x, y = inputs[i], targets[i]
                output = self.propagate(x)
                loss = self.get_sum_squared_error(output[self.num_layers-1], targets)
                error.append(loss)
                self.back_propagate(output, y)

            

    def process_layer(self, inputs, layer_id):
        values, num_neurons, num_inputs = np.array([]), self.layers[layer_id+1], self.layers[layer_id]
        for neuron_id in range(num_neurons):        # parse through each neuron of layer
            value = self.process(inputs, self.weights[layer_id][neuron_id*num_inputs:neuron_id*num_inputs+num_inputs])
            values = np.append(values, [value])
        return values

    
    def propagate(self, inputs):
        values = {}
        for layer_id in range(self.num_layers):     # parse through each layer
            inputs = self.process_layer(inputs, layer_id)
            values[layer_id] = inputs
        return values