from nn_aux import relu, timeit, get_error, get_delta
import numpy as np

class NeuralNetwork():
    """
    Neural Network
    """
    def __init__(self, num_inputs):
        """
        Create an empty Neural Network with num_inputs sensory neurons
        Add layers to the Network with add_layer(num_neurons) the layer added last will be considered the output Layer
        """
        np.random.seed(4)
        self.layer_neuron_counts = [num_inputs]
        self.neuron_memories = []
        self.weights = []
        self.adjustments = {}

    def __repr__(self):
        repr_str = 'self.layer_neuron_counts: ' + str(self.layer_neuron_counts) + '\n'
        repr_str += 'self.neuron_memories: \n'
        for key in range(len(self.neuron_memories)):
            repr_str += str(self.neuron_memories[key]) + '\n'
        repr_str += 'self.weights: \n'
        for key in range(len(self.weights)):
            repr_str += str(self.weights[key]) + '\n'
        return repr_str

    def __make_layer(self, shape):
        return (2 * np.random.random(shape) - 1)

    def add_layer(self, layer_size):
        last_layer = self.layer_neuron_counts[-1]
        self.weights.append(self.__make_layer((layer_size, last_layer)))
        self.adjustments[len(self.layer_neuron_counts)] = np.zeros((layer_size, last_layer))
        self.layer_neuron_counts.append(layer_size)

    def process_layer(self, inputs, layer_id):
        return relu(np.dot(inputs, self.weights[layer_id-1].T))

    def propagate(self, inputs):
        memories = [inputs]
        for layer_id in range(1, len(self.layer_neuron_counts)):     # parse through each layer
            inputs = self.process_layer(inputs, layer_id)
            memories.append(inputs)
        self.neuron_memories = memories
        return memories

    def gradient_descent(self, batch_size, learning_rate=0.001):
        # Calculate partial derivative and take a step in that direction
        num_layers = len(self.layer_neuron_counts)
        for layer in range(1, num_layers):
            partial_d = (1/batch_size) * self.adjustments[layer]
            self.weights[layer][:-1, :] += learning_rate * -partial_d
            self.weights[layer][-1, :] += learning_rate * -partial_d[-1, :]

    def back_propagate(self, targets):
        deltas = {}
        num_layers = len(self.layer_neuron_counts)
        deltas[num_layers-1] = get_delta(self.neuron_memories[-1], targets)
        for weight_set in reversed(range(1, num_layers-1)):
            prev_deltas = deltas[weight_set+1]
            current_layer_values = self.neuron_memories[weight_set]
            current_weights = self.weights[weight_set]
            dot = np.dot(current_weights.T, prev_deltas)
            relu_derivative = relu(current_layer_values, True)
            deltas[weight_set] = np.multiply(dot, relu_derivative)
        for layer in range(1, num_layers):
            self.adjustments[layer] += np.dot(deltas[layer], self.neuron_memories[layer]).T

    def train(self, inputs, targets, iterations=2):
        for _ in range(iterations):
            error = []
            for i in range(len(inputs)):
                x, y = inputs[i], targets[i]
                self.propagate(x)
                error.append(get_error(self.neuron_memories[-1], y))
                self.back_propagate(y)
            self.gradient_descent(i)