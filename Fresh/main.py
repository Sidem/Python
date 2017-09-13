from NeuralNetwork import NeuralNetwork
import numpy as np

NN = NeuralNetwork(2)
NN.add_layer(4)
NN.add_layer(4)
NN.add_layer(2)
INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
OUTPUTS = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
print('UNTRAINED: ', NN.think(INPUTS[2]))
NN.train(INPUTS, OUTPUTS, 50)
print('50: ', NN.think(INPUTS[2]))
NN.train(INPUTS, OUTPUTS, 500)
print('550: ', NN.think(INPUTS[2]))
NN.train(INPUTS, OUTPUTS, 500)
print('5500: ', NN.think(INPUTS[2]))
#NN.propagate(INPUTS[2])
#print(NN)
#print(NN)