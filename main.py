from NeuralNet import NeuralNet
import numpy as np


NN = NeuralNet(3)
#print(NN)
NN.add_layer(5) #1
NN.add_layer(6) #2
NN.add_layer(5) #3
NN.add_layer(1) #4 OUTPUT
INPUTS = np.array([np.array([1, 0, 1]).T])
OUTPUTS = np.array([[1]])
NN.train(INPUTS, OUTPUTS, 1)



#NN.train(INPUTS, OUTPUTS, 5)
