from NeuralNet import NeuralNet
import numpy as np


NN = NeuralNet(3)
#print(NN)
NN.addLayer(3)
NN.addLayer(4)
NN.addLayer(5)
NN.addLayer(6)
inputArray = np.array([0, 0, 1]).T
NN.propagate(inputArray)