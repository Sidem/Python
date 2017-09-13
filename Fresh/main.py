from NeuralNetwork import NeuralNetwork
import numpy as np

NN = NeuralNetwork(2)
NN.add_layer(3)
NN.add_layer(3)
NN.add_layer(2)
INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
OUTPUTS = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
NN.train(INPUTS, OUTPUTS)
print(NN.adjustments)
#print(NN)