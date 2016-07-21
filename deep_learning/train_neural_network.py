from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import sys

pickle_file = sys.argv[1]

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	train_dataset = save['train_dataset']
	train_labels = save['train_labels']
	test_dataset = save['test_dataset']
	test_labels = save['test_labels']
	del save
	print('Training set', train_dataset.shape, train_labels.shape)
	print('Test set', test_dataset.shape, test_labels.shape)

#Reformat labels to 1-hot encoding for training
num_labels = 2
train_dataset = train_dataset.astype(np.float32)
train_labels = (np.arange(num_labels) == train_labels[:, None]).astype(np.float32)
test_dataset = test_dataset.astype(np.float32)
test_labels = (np.arange(num_labels) == test_labels[:, None]).astype(np.float32)

#Train Neural Network
m = train_dataset.shape[0]
n = train_dataset.shape[1]
hidden_units = 1024
graph = tf.Graph()
with graph.as_default():

	#Input data
	tf_train_dataset = tf.constant(train_dataset)
	tf_train_labels = tf.constant(train_labels)
	tf_test_dataset = tf.constant(test_dataset)
	tf_test_labels = tf.constant(test_labels)

	#Parameters to be trained
	weights1 = tf.Variable(tf.truncated_normal([n, hidden_units]))
	biases1 = tf.Variable(tf.zeros([hidden_units]))
	weights2 = tf.Variable(tf.truncated_normal([hidden_units, num_labels]))
	biases2 = tf.Variable(tf.zeros([num_labels]))

	#Training computation
	logits_hidden = tf.matmul(tf_train_dataset, weights1) + biases1
	logits_hidden = tf.sigmoid(logits_hidden)
	logits = tf.matmul(logits_hidden, weights2) + biases2
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

	#Optimizer
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

	#Predictions for the training and test data
	train_prediction = tf.nn.softmax(logits)
	test_logits_hidden = tf.matmul(tf_test_dataset, weights1) + biases1
	test_logits_hidden = tf.sigmoid(test_logits_hidden)
	test_prediction = tf.nn.softmax(tf.matmul(test_logits_hidden, weights2) + biases2)

num_steps = 8001

def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

with tf.Session(graph=graph) as session:
	#Initialize variables
	tf.initialize_all_variables().run()
	print('Initialized')
	for step in range(num_steps):
		#Run the computations
		_, l, predictions = session.run([optimizer, loss, train_prediction])
		if(step % 100 == 0):
			print('Loss at step %d: %f' % (step, l))
			print('Training accuracy: %.1f%%' % accuracy(predictions, train_labels))
			print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))