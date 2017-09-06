from NeuralNet import NeuralNet
import numpy as np


NN = NeuralNet(2)
#print(NN)
NN.addLayer(3)
NN.addLayer(1)
inputArray = np.array([0, 0]).T
NN.propagate(inputArray)