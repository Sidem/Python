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

def relu_derivative(x):
    if x > 0:
        return 1
    else:
        if LEAKY_RELU:
            return 0.01
        else:
            return 0

def relu(x, derivative=False):
    if derivative:
        return np.fromiter((relu_derivative(xi) for xi in x), x.dtype)
    else:
        if LEAKY_RELU:
            return np.maximum(x, x*0.01, x)
        else:
            return np.maximum(x, 0, x)
            

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

    def __make_layer(self, shape):
        return (2 * np.random.random(shape) - 1)

    def add_layer(self, layer_size):
        if self.num_layers == 0:
            self.weights.append(self.__make_layer((layer_size, self.num_inputs)))
        else:
            self.weights.append(self.__make_layer((layer_size, self.layers[self.num_layers]))) 
        self.layers[self.num_layers+1] = layer_size
        self.num_layers += 1

    def back_propagate(self, output, targets):
        deltas = {}
        # Delta of output Layer
        print('num_layers',self.num_layers)
        print('output', output)
        print('targets', targets)
        deltas[self.num_layers-1] = self.get_error(output[self.num_layers-1], targets)
        for layer_id in reversed(range(self.num_layers-1)):
            a_val = output[layer_id]
            weights = self.weights[layer_id][:-1, :]
            prev_deltas = deltas[layer_id+1]
            print('WEIGHT',weights)
            print('PREV_DELTAS',prev_deltas)
            dot = np.dot(weights, prev_deltas)
            aval_relu = relu(a_val, True)
            print('AVAL',a_val)
            print('DOT',dot)
            print('AVAL_RELU',aval_relu)
            deltas[layer_id] = np.multiply(dot, aval_relu)

    def train(self, inputs, targets, iterations):
        error = []
        for _ in range(iterations):
            for i in range(len(inputs)):
                x, y = inputs[i], targets[i]
                output = self.propagate(x)
                loss = self.get_sum_squared_error(output[self.num_layers-1], targets)
                error.append(loss)
                self.back_propagate(output, y)

    def process(self, inputs, layer):
        return relu(np.array([np.dot(inputs, layer)]))        

    def process_layer(self, inputs, layer_id):
        values, num_neurons, num_inputs = np.array([]), self.layers[layer_id+1], self.layers[layer_id]
        for neuron_id in range(num_neurons):        # parse through each neuron of layer
            value = self.process(inputs, self.weights[layer_id][neuron_id])
            values = np.append(values, [value])
        return values

    @timeit
    def propagate(self, inputs):
        values = {}
        for layer_id in range(self.num_layers):     # parse through each layer
            inputs = self.process_layer(inputs, layer_id)
            values[layer_id] = inputs
        self.values = values
        return values