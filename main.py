from NeuralNet import NeuralNet
from numpy import exp, array, random, dot

neural_network = NeuralNet(3,3)
print('Random starting synaptic weights: \n'+str(neural_network.synaptic_weights))

training_set_inputs = array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
training_set_outputs = array([[0,1,0,1,0,1,0,1]]).T

neural_network.train(training_set_inputs, training_set_outputs, 100000)

print('synaptic weights after training: \n'+str(neural_network.synaptic_weights))

test_input = array([[1,1,1]])
print('testing input: '+str(test_input))
test_output = neural_network.think(test_input)
print('resulting output: '+str(format(test_output[0][0], '.12f')))

test_input = array([[1,0,1]])
print('testing input: '+str(test_input))
test_output = neural_network.think(test_input)
print('resulting output: '+str(format(test_output[0][0], '.12f')))

test_input = array([[1,1,0]])
print('testing input: '+str(test_input))
test_output = neural_network.think(test_input)
print('resulting output: '+str(format(test_output[0][0], '.12f')))

test_input = array([[0,1,1]])
print('testing input: '+str(test_input))
test_output = neural_network.think(test_input)
print('resulting output: '+str(format(test_output[0][0], '.12f')))