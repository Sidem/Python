import os
from time import time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("tmp/data/", one_hot=True)

layer_sizes = [784, 500, 500, 500, 10]
batch_size = 100
epochs_no = 10

# input feature size = 28x28 pixels = 784
x = tf.placeholder('float', [None, layer_sizes[0]])
y = tf.placeholder('float')

def propagate(data, layers):
    current_layer_result = data
    for i in range(len(layers)):
        current_layer_result = tf.add(tf.matmul(current_layer_result, layers[i]['weights']), layers[i]['biases'])
        current_layer_result = tf.nn.relu(current_layer_result)
    return current_layer_result

def construct_layers(layer_sizes):
    layers = []
    for i in range(len(layer_sizes)-1):
        layers.append({'weights': tf.Variable(tf.random_normal([layer_sizes[i], layer_sizes[i+1]])),
                       'biases': tf.Variable(tf.random_normal([layer_sizes[i+1]]))})
    return layers

def neural_network_model(data):
  # input_data * weights + biases
  layers = construct_layers(layer_sizes)
  output = propagate(data, layers)
  return output

def train_neural_network(x):
  prediction = neural_network_model(x)
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))   # v1.0 changes
  # optimizer value = 0.001, Adam similar to SGD
  optimizer = tf.train.AdamOptimizer().minimize(cost)
 
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   # v1.0 changes
  
    # training
    for epoch in range(epochs_no):
      epoch_loss = 0
      for _ in range(int(mnist.train.num_examples/batch_size)):
        epoch_x, epoch_y = mnist.train.next_batch(batch_size)
        _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
          # code that optimizes the weights & biases
        epoch_loss += c
      print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', epoch_loss)
  
    # testing
    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

before = time()  
train_neural_network(x)
after = time()
print('time elapsed: ', after - before, 'seconds')