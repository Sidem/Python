import numpy as np

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
        self.weights.append(2 * np.random.random((num_inputs, 1)) - 1)

    def __repr__(self):
        repr_str = ""
        for layer in self.weights:
            repr_str += str(layer) + "\n"
        return repr_str
    
    def addLayer(self, layer_size):
        self.weights.append(2 * np.random.random((layer_size, 1)) - 1)

    def train(self, inputs, outputs, iterations):
        for i in range(iterations):
            output = self.think(inputs)
            error = outputs - output
            adjustment = np.dot(inputs.T, error * relu(output, True))
            self.weights += adjustment
                   
    def think(self, inputs, mode=True):
        """
        args:
            inputs: array of values
            mode:   True    -> applies relu function
                    False   -> applies sigmoid function
        return:
            resulting single value
        """
        if mode:
            return relu(np.dot(inputs, self.weights))
        return sigmoid(np.dot(inputs, self.weights))