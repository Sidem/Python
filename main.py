from NeuralNet import NeuralNet
import numpy as np


NN = NeuralNet(3)
#print(NN)
NN.add_layer(5) #0
NN.add_layer(5) #1
NN.add_layer(5) #2
NN.add_layer(1) #3 OUTPUT
INPUTS = np.array([np.array([1, 1, 1]).T])
OUTPUTS = np.array([[1]])
NN.train(INPUTS, OUTPUTS, 5)
