from numpy import exp, array, random, dot

class NeuralNet():
    def __sigmoid(self, x, derivative=False):
        if derivative == True:
            return x * (1 - x)
        return 1 / (1 + exp(-x))
    
    def __relu(self, layer_inputs, derivative=False):
        for i in range(len(layer_inputs)):
            if layer_inputs[i] > 0:
                if derivative:
                    layer_inputs[i] = 1
                else:
                    pass
            else:
                layer_inputs[i] = 0
        return layer_inputs

    def __repr__(self):
        return str(self.synaptic_weights)
    
    def __init__(self, num_inputs, num_layers):
        random.seed(1)
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.synaptic_weights = 2 * random.random((num_inputs, 1)) - 1

    def propagate(self):
        pass
    
    def b_propagate(self):
        pass
    
    def train(self, inputs, outputs, iterations):
        for i in range(iterations):
            output = self.think(inputs)
            error = outputs - output
            adjustment = dot(inputs.T, error * self.__relu(output, True))
            self.synaptic_weights += adjustment
            
            
    def think(self, inputs, relu=True):
        if relu:
            return self.__relu(dot(inputs, self.synaptic_weights))
        return self.__sigmoid(dot(inputs, self.synaptic_weights))