# perceptron_lab.py -- Implement a single perceptron (artificial neuron) and train it to solve a logical AND operation.
#
# YOUR NAME HERE
# Caitrin Eaton
# Machine Learning for Visual Thinkers
# Fall 2020

import numpy as np
import matplotlib.pyplot as plt


def sigmoid_activation( X, W, threshold=0 ):
	''' A logistic sigmoid activation function

	INPUTS
	X -- (n,m+1) ndarray of inputs, in which each row represents 1 sample + normal homogeneous coordinate, and each column represents 1 feature
	W -- (m+1,1) ndarray of weights for each input feature as well as the bias term (intercept)
	threshold -- float, determines how strongly residuals impact weights during each iteration of training.

	OUTPUTS
	activation -- (n,1) ndarray, the perceptron's output in response to each sample (row) in X
	'''
	weighted_sum = X @ W
	activation = 1 / (1 + np.exp( -(weighted_sum - threshold) ))
	return activation


def relu_activation( X, W, threshold=0 ):
	''' A rectified linear ("ReLU") activation function

	INPUTS
	X -- (n,m+1) ndarray of inputs, in which each row represents 1 sample + normal homogeneous coordinate, and each column represents 1 feature
	W -- (m+1,1) ndarray of weights for each input feature as well as the bias term (intercept)
	threshold -- float, determines how strongly residuals impact weights during each iteration of training.

	OUTPUTS
	activation -- (n,1) ndarray, the perceptron's output in response to each sample (row) in X
	'''
	n = X.shape[0]
	zero = np.zeros( (n,1) )
	weighted_sum = np.hstack((X @ W, zero))
	activation = np.max( weighted_sum, axis=1 ) 
	return activation


def step_activation( X, W, threshold=0 ):
	''' A step activation function.

	INPUTS
	X -- (n,m+1) ndarray of inputs, in which each row represents 1 sample + normal homogeneous coordinate, and each column represents 1 feature
	W -- (m+1,1) ndarray of weights for each input feature as well as the bias term (intercept)
	threshold -- float, determines how strongly residuals impact weights during each iteration of training.

	OUTPUTS
	activation -- (n,1) ndarray, the perceptron's output in response to each sample (row) in X
	'''
	n = X.shape[0]
	weighted_sum = X @ W
	activation = np.zeros((n,1))
	activation[ weighted_sum > threshold ] = 1.0
	return activation


def train_perceptron( X, Y, learning_rate=0.5, threshold=0 ):
	''' Train a perceptron to predict the training targets Y given training inputs X. 

	INPUTS
	X -- (n,m) ndarray of training inputs, in which each row represents 1 sample and each column represents 1 feature
	Y -- (n,1) ndarray of training targets
	learning_rate -- float, determines how strongly residuals impact weights during each iteration of training.

	OUTPUTS
	W -- (m+1,) ndarray of weights for each column of X + the bias term (intercept) 
	Y_pred -- (n,1) ndarray of predictions for each sample (row) of X
	mse -- float, mean squared error
	'''

	# Create a figure window for tracking mse over time
	fig = plt.figure()
	plt.title( "Perceptron training" )
	plt.xlabel( "epoch" )
	plt.ylabel( "MSE" )
	plt.grid( True )

	# TODO: Add a normal homogeneous coordinate to X
	n = X.shape[0]
	ones = np.ones((n, 1))
	A = np.hstack((X, ones))

	# TODO: Initialize weights to small random numbers
	m = X.shape[1]
	W = np.random.random((m + 1, 1))

	# TODO: Loop over training set until error is acceptably small, or iteration cap is reached	
	tolerance = 0.001
	max_epoch = 100
	epoch = 0
	mse = tolerance * 10
	Y_pred = np.zeros((n, 1))

	while epoch < max_epoch and tolerance < mse:
		epoch += 1
		for i in range(n):
			sample = A[i, :].reshape((1, m + 1))
			#Y_pred[i, 0] = step_activation(sample, W, threshold)
			Y_pred[i, 0] = sigmoid_activation(sample, W, threshold)
			#Y_pred[i, 0] = relu_activation(sample, W, threshold)
			residual = Y_pred[i, 0] - Y[i, 0]
			W = W - sample.T * residual * learning_rate

		mse = np.mean((Y_pred - Y)**2, axis=0)
		plt.plot(epoch, mse, 'ok')
		plt.pause(0.001)

	return W, Y_pred, mse


def test_logical_and():
	''' Train a perceptron to perform a logical AND operation. '''
	truth_table = np.array( [[0,0,0], [0,1,0], [1,0,0], [1,1,1]] )
	n = truth_table.shape[0]
	X = truth_table[:, 0:2]
	Y = truth_table[:, 2].reshape((n,1))
	learning_rate = 0.5
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate )

	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "------------------" )
	print( " X0  X1 Y  Y* " )
	print( "------------------" )
	print( np.hstack((truth_table, Y_pred)) )
	print( "------------------" )
	
	plt.title( "AND" )
	plt.show()


def test_logical_or():
	''' Train a perceptron to perform a logical OR operation. '''
	truth_table = np.array( [[0,0,0], [0,1,1], [1,0,1], [1,1,1]] )
	n = truth_table.shape[0]
	X = truth_table[:, 0:2]
	Y = truth_table[:, 2].reshape((n,1))
	learning_rate = 0.5
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate )
	
	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "------------------" )
	print( " X0  X1 Y  Y* " )
	print( "------------------" )
	print( np.hstack((truth_table, Y_pred)) )
	print( "------------------" )

	plt.title( "OR" )
	plt.show()


def test_logical_xor():
	''' Train a perceptron to perform a logical XOR operation. '''
	truth_table = np.array( [[0,0,0], [0,1,1], [1,0,1], [1,1,0]] )
	n = truth_table.shape[0]
	X = truth_table[:, 0:2]
	Y = truth_table[:, 2].reshape((n,1))
	learning_rate = 0.5
	W, Y_pred, mse = train_perceptron( X, Y, learning_rate )
	
	# Display results in the temrinal
	print( "\nWeights:", W.T )
	print( "MSE:", mse )
	print( "------------------" )
	print( " X0  X1 Y  Y* " )
	print( "------------------" )
	print( np.hstack((truth_table, Y_pred)) )
	print( "------------------" )

	plt.title( "XOR" )
	plt.show()


if __name__=="__main__":
	test_logical_and()
	#test_logical_or()
	#test_logical_xor()