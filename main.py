from NeuralNet import NeuralNet
import numpy as np


NN = NeuralNet(3)
#print(NN)
NN.add_layer(4) #1
NN.add_layer(5) #2
NN.add_layer(3) #3 OUTPUT
#print(NN.weights[0])

i_0 = np.array([0, 0, 0]).T
i_1 = np.array([0, 0, 1]).T
i_2 = np.array([0, 1, 0]).T
i_3 = np.array([0, 1, 1]).T
i_4 = np.array([1, 0, 0]).T
i_5 = np.array([1, 0, 1]).T
i_6 = np.array([1, 1, 0]).T
i_7 = np.array([1, 1, 1]).T

o_0 = np.array([0, 0, 1]).T
o_1 = np.array([0, 1, 0]).T
o_2 = np.array([0, 1, 1]).T
o_3 = np.array([1, 0, 0]).T
o_4 = np.array([1, 0, 1]).T
o_5 = np.array([1, 1, 0]).T
o_6 = np.array([1, 1, 1]).T
o_7 = np.array([0, 0, 0]).T

INPUTS = np.array([i_0, i_1, i_2, i_3, i_4, i_5, i_6, i_7])
OUTPUTS = np.array([o_0, o_1, o_2, o_3, o_4, o_5, o_6, o_7])
#NN.train(INPUTS, OUTPUTS, 1)
NN.propagate(i_0)
print(NN)