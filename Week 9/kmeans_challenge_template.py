# kmeans_challenge.py -- A homebrewed implementation of K-means clustering.
#
# Caitrin Eaton
# Machine Learning for Visual Thinkers
# Week 9: Clustering
# Fall 2020

import os
import numpy as np
import matplotlib.pyplot as plt


def plot_clusters( X, clustering, means, headers, ax=None ):
	'''Plots the 2D projection of X onto its first 2 dimensions using unique colors for each cluster. 
	
	INPUTS:
	X -- (n,m) ndarray that represents the dataset, where rows are samples and columns are features
	clustering -- (n,1) ndarray of cluster assignments, each in the range [0, k-1]
	means -- (k,m) ndarray of cluster means
	headers -- a list of feature names (strings), the names of the columns of X

	OUTPUTS:
	ax -- a reference to the axis on which the clusters are plotted
	'''
	
	# Determine how many clusters there are, and what color each will be
	k = len( np.unique(clustering) )
	colors = plt.cm.viridis( np.linspace(0,1,k) )

	# Initialize the axes
	if ax == None:
		fig, ax = plt.subplots() # no axis supplied, make one
	else:
		ax.clear()	# an axis was supplied, make sure it's empty
	ax.set_xlabel( headers[0] )
	ax.set_ylabel( headers[1] )
	ax.set_title( f"K-Means clustering, K={k}" )
	ax.grid(True)
		
	# Plot each cluster in its own unique color
	for cluster_id in range(k):
		
		# Pull out the cluster's members: just the rows of X in cluster_id
		members = (clustering == cluster_id).flatten()
		
		# Plot this cluster's members in this cluster's unique color
		plt.plot(X[members, 0], X[members, 1], 'o', alpha = 0.5, markerfacecolor = colors[cluster_id], markeredgecolor = colors[cluster_id], label = str(cluster_id))
		
		# Plot this cluster's mean (making it a shape distinct from data points, e.g. a larger diamond)
		plt.plot(means[cluster_id, 0], means[cluster_id, 1], 'd', markerfacecolor = colors[cluster_id], markeredgecolor = 'w', linewidth = 2, markersize = 15)
	
	return ax


def kmeans( X, k, headers ):
	''' Partition dataset X into k clusters using the K-means clustering algorithm. 
	
	INPUT
	X -- (n,m) ndarray that represents the dataset, where rows are samples and columns are features
	k -- int, the number of clusters
	headers -- a list of feature names (strings), the names of the columns of X

	OUTPUT
	clustering -- (n,1) ndarray indicating the cluster labels in the range [0, k-1]
	means -- (k,m) ndarray representing the mean of each cluster
	'''
	# TODO: Fill in the K-means algorithm here	
	# Initialize K guesses regarding the means

	n = X.shape[0]
	m = X.shape[1]
	mins = np.min(X, axis = 0)
	maxs = np.max(X, axis = 0)
	ranges = maxs - mins
	means = np.random.random((k, m)) * ranges + mins

	# While not done, place each point in the cluster with the nearest mean
	# (done when no point changes clusters)
	ax = None
	clustering_old = np.ones((n,)) * -2
	clustering = np.ones((n,)) * -1
	dist = np.zeros((n, k))
	iteration = 0

	while 10**(-10) < np.sum(np.abs(clustering_old - clustering)) and iteration < 100:
		# So that we can tell later if any points have changed clusters
		clustering_old = clustering.copy()
		iteration += 1

		# Compute the distance of each point to all the means
		for cluster_id in range(k):
			# Compute the distance of each point to this particular mean
			dist[:, cluster_id] = np.linalg.norm(X - means[cluster_id, :], axis = 1)

		clustering = np.argsort(dist, axis = 1)[:, 0]

		for cluster_id in range(k):
			members = clustering == cluster_id
			means[cluster_id, :] = np.mean(X[members, :], axis = 0)

		ax = plot_clusters(X, clustering, means, headers, ax)
		ax.set_title(f"K-Means, K={k}, iteration {iteration}")
		plt.pause(0.1)

	return clustering, means


def read_file( filename, header_row=True, delimiter="," ):
	''' Read in the data from a file.

	INPUT:
	filename -- string representing the name of a file in the "../data/" directory
	header_row -- bool indicating whether or not the first line of the file contains metadata (column headers)
	delimiter -- string representing the character(s) separating columns of data

	OUTPUT:
	data -- (n,m) ndarray of data from the specified file, assuming 1 row per sample and 1 column per feature
	headers -- list of length m representing the name of each feature (column)
	'''
	
	# Windows is kind of a jerk about filepaths. My relative filepath didn't
	# work until I joined it with the current directory's absolute path.
	current_directory = os.path.dirname(__file__)
	filepath = os.path.join(current_directory, "..", "data", filename)
	print( "\nfilepath:", filepath )

	# Read the data in, skipping the metadata in 1st row
	if header_row == True:
		data = np.genfromtxt( filepath, delimiter=delimiter, skip_header=1 )
	else:
		data = np.genfromtxt( filepath, delimiter=delimiter )

	# Read headers from the 1st row with plain vanilla Python file handling (without Numpy)
	if header_row == True:
		in_file = open( filepath )
		headers = in_file.readline().split( delimiter )
		in_file.close()
	else:
		# No metadata in the file, so call each column "Xi", where i is an integer
		m = data.shape[1]
		headers = []
		for i in range(m):
			headers.append( "X" + str(i) )
	
	return data, headers


def cluster_analysis( filename, x_col, y_col, k, class_col=None ):
	''' Apply K-means clustering to the specified dataset.'''
	
	# Read in the dataset
	X, headers = read_file( filename )
	n = X.shape[0]
	m = X.shape[1]
	
	# Remove the class label from the dataset so that it doesn't dominate the cluster analysis
	if class_col != None:
		class_labels = X[:, class_col]
		keepers = list(range(m))
		keepers.pop( class_col )
		X = X[:, keepers]
		class_header = headers[ class_col ]
		headers.pop( class_col )
		m = X.shape[1]

	# Visualize raw data
	print(np.mean(X, axis=0).reshape((1,m)))
	ax = plot_clusters( X, np.zeros((n,1)), np.mean(X, axis=0).reshape((1,m)), headers, ax=None )
	ax.set_title( "Raw Data" )
		
	# Compute K-Means
	clustering, means = kmeans( X, k,headers )
	print(means.shape)
	# Visualize all K final clusters
	ax = plot_clusters( X, clustering, means, headers, ax=None )
	ax.set_xlabel( headers[x_col] )
	ax.set_ylabel( headers[y_col] )
	plt.tight_layout()


if __name__=="__main__":
	cluster_analysis( "iris_preproc.csv", x_col=0, y_col=1, k=3, class_col=4 )
	#plt.show()