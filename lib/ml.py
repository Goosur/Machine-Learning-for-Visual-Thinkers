# ml.py -- functions for machine learning projects
# 
# Devon Gardner
# Machine Learning for Visual Thinkers
# 11/03/2020

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os, sys


def knn(X_tra, X_test, k):#, headers):
	''' Partition dataset X into k clusters using the K-means clustering algorithm. 
	
	INPUT
	X -- (n,m) ndarray that represents the dataset, where rows are samples and columns are features
	k -- int, the number of clusters
	headers -- a list of feature names (strings), the names of the columns of X

	OUTPUT
	clusters -- (n,1) ndarray indicating the cluster labels in the range [0, k-1]
	means -- (k,m) ndarray representing the mean of each cluster
	'''
	ax = None
	c_pred = []

	for i in range(X_test.shape[0]):
		# Pull sample
		s = pd.DataFrame(X_test.iloc[i, :]).T
		
		# Calculate euclidean distance between sample and all other points
		D = pd.Series(np.linalg.norm(np.subtract(X_tra.iloc[:, :-1], s), axis = 1), index=[X_tra.iloc[:, :-1].index], name='Distance')
		ids = D.argsort()
		C_sorted = X_tra.iloc[ids, -1]
		unique, counts = np.unique(C_sorted[0:k], return_counts=True)
		c_pred.append(unique[np.argmax(counts)])

	return pd.Series(c_pred, index=X_test.index, name=X_tra.columns[-1])

def vis_clusters(X, clusters, means, headers, ax=None):
	""" Apply clusters to dataset X and plot alongside means.

	Args:
		X (ndarray): (n,m) ndarray that represents the dataset
		clusters (ndarray): (1,n) ndarray indicating cluster labels
		means (ndarray): (k,m) representing cluster means
		ax (axis, optional): Axis to plot. Defaults to None.
		headers (list): a list of feature names (strings), the names of the columns of X

	Returns:
		axis: Axis that was plotted on.
	"""
	# Determine how many clusters there are, and what color each will be
	k = len( pd.unique(clusters) )
	colors = plt.cm.viridis( np.linspace(0,1,k) )
		
	# Initialize the axes
	if ax == None:
		fig, ax = plt.subplots() # no axis supplied, make one
	else:
		ax.clear()	# an axis was supplied, make sure it's empty
	ax.set_xlabel(headers[0])
	ax.set_ylabel(headers[1])
	ax.set_title( f"K-Means clusters, K={k}" )
	ax.grid(True)
	
	# Plot each cluster in its own unique color
	for cluster in range(k):
		
		# Pull out the cluster's members: just the rows of X in cluster_id
		members = clusters == cluster
	
		# Plot this cluster's members in this cluster's unique color
		plt.plot(X[members, 0], X[members, 1], 'o', alpha=0.5, markerfacecolor=colors[cluster], markeredgecolor=colors[cluster], label=cluster)

		# Plot this cluster's mean (making it a shape distinct from data points, e.g. a larger diamond)
		plt.plot(means[cluster, 0], means[cluster, 1], 'd', markerfacecolor=colors[cluster], markeredgecolor='w', linewidth=2, markersize=15)

	return ax


def kmeans(X, k, headers):
	''' Partition dataset X into k clusters using the K-means clustering algorithm. 
	
	INPUT
	X -- (n,m) ndarray that represents the dataset, where rows are samples and columns are features
	k -- int, the number of clusters
	headers -- a list of feature names (strings), the names of the columns of X

	OUTPUT
	clusters -- (n,1) ndarray indicating the cluster labels in the range [0, k-1]
	means -- (k,m) ndarray representing the mean of each cluster
	'''
	# Initialize K guesses regarding the means
	n = X.shape[0]
	m = X.shape[1]
	mins = np.min(X, axis=0)
	maxs = np.max(X, axis=0)
	ranges = maxs - mins
	means = np.random.random((k, m)) * ranges + mins

	# While not done, place each point in the cluster with the nearest mean
	# (done when no point changes clusters)
	ax = None
	clusters_old = np.ones((n,)) * -2
	clusters = np.ones((n,)) * -1
	dist = np.zeros((n, k))
	iteration = 0

	while 10**(-10) < np.sum(np.abs(clusters_old - clusters)) and iteration < 100:
		# So that we can tell later if any points have changed clusters
		clusters_old = clusters.copy()
		iteration += 1

		# Compute the distance of each point to all the means
		for cluster_id in range(k):
			# Compute the distance of each point to this particular mean
			dist[:, cluster_id] = np.linalg.norm(X - means[cluster_id, :], axis = 1)

		clusters = np.argsort(dist, axis = 1)[:, 0]

		for cluster_id in range(k):
			members = clusters == cluster_id
			means[cluster_id, :] = np.mean(X[members, :], axis = 0)

		# ax = vis_clusters(X, clusters, means, headers, ax)
		# ax.set_title(f"K-Means, K={k}, iteration {iteration}")
		# plt.pause(0.1)

	return clusters, means


def pc_heatmap( P, info_retention ):
	''' Visualize principal components (eigenvectors) as a heatmap. 
	
	INPUT:
	P -- (m,m) ndarray of principal components (eigenvectors)
	info_retention -- (m,) ndarray of accumulated scaled eigenvectors: the % info retained by all PCs
	
	OUTPUT: 
	None
	'''
	fig, ax = plt.subplots()

	ax = sns.heatmap(P.abs(), cmap = 'bone', vmin = 0, vmax = 1, fmt='.3f', square = True)
	ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right')
	ax.set_title('PCs')


def scree_plot(eigenvals):
	"""Visualize information retention per eigenvector.
	
	INPUT:	
	eigenvals -- (d,) ndarray of scaled eigenvalues.
	
	OUTPUT:
	info_retention -- (d,) ndarray of accumulated information retained by multiple eigenvectors.  """
			
	# Visaulize individual information retention per eigenvector (eigenvalues)
	fig, ax = plt.subplots(2, 1)
	ax[0].plot(eigenvals, '-o', linewidth=2, markersize=5, markerfacecolor="w")
	ax[0].set_ylim([-0.1, 1.1])
	ax[0].set_title("Information retained by individual PCs")
	ax[0].grid(True)
	
	# Visualize accumulated information retained by multiple eigenvectors
	info_retention = eigenvals.cumsum()
	ax[1].plot(info_retention, '-o', linewidth=2, markersize=5, markerfacecolor="w")
	ax[1].set_ylim([-0.1, 1.1])
	ax[1].set_title("Cumulative information retained by all PCs")
	ax[1].grid(True)
		
	return info_retention


def pca_cov( X ):
	"""Perform Principal Components Analysis (PCA) using the covariance matrix to identify principal components 
	(eigenvectors) and their scaled eigenvalues (which measure how much information each PC represents).
	
	INPUT:
	X -- (n,m) ndarray representing the dataset (observations), assuming one datum per row and one column per feature. 
			Must already be centered, so that the mean is at zero. Usually Z-score normalized. 
	
	OUTPUT:
	Y -- (n,m) ndarray representing rotated dataset (Y), 
	P -- (m,m) ndarray representing principal components (columns of P), a.k.a. eigenvectors
	e_scaled -- (m,) ndarray of scaled eigenvalues, which measure info retained along each corresponding PC """
	
	# Calculate principal components and eigenvalues using covariance.
	C = np.cov(X, rowvar=False)
	(e, P) = np.linalg.eig(C)

	# Pandafy e_scaled and P
	e = pd.Series(e)
	P = pd.DataFrame(P)

	# Generate pandas labels
	for i in P.columns:
		P.rename(columns = {i: 'P' + str(i)}, inplace = True)
	P.index = X.columns
	for i in e.index:
		e.rename({i: 'e' + str(i)}, inplace = True)

	# Sort principal components in order of descending eigenvalues
	order = e.argsort(axis = 'index')[::-1]
	e = e.iloc[order]
	P = P.iloc[:, order]

	# Scale eigenvalues to calculate the percent info retained along each PC
	e_scaled = e / e.sum()

	# Rotate data onto the principal components
	Y = X.dot(P.to_numpy())
	Y.columns = P.columns

	return (Y, P, e_scaled)


def heatmap(cov, title, color_bar_label):
	'''Generate heatmap of given data'''
	fig, ax = plt.subplots()
	
	ax = sns.heatmap(cov, cmap = 'viridis', annot = True, fmt='.3f', square = True, cbar_kws = {'label': color_bar_label})
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
	ax.set_title(title)


def center(X):
	"""Center given dataset using mean.

	Args:
		X (DataFrame): (n, m) DataFrame of raw data, assumed to contain 1 row per sample and 1 column per feature.

	Returns:
		DataFrame: DataFrames returned contain the centered data and the means. 
	"""
	X_mean = X.mean(axis = 0)
	X_centered = pd.DataFrame(X - X_mean)

	return X_centered, X_mean


def z_norm(X):
	''' Normalize the dataset X by Z-score: subtract the mean and divide by the standard deviation.

	INPUT:
	X -- (n,m) ndarray of raw data, assumed to contain 1 row per sample and 1 column per feature.

	OUTPUT:
	X_norm -- (n,m) ndarray of Z-score normalized data
	X_mean -- (m,) ndarray of the means of the features (columns) of the raw dataset X
	X_std -- (m,) ndarray of the standard deviations of the features (columns) of the raw dataset X
	'''

	X_mean = X.mean(axis = 0)
	X_std = X.std(axis = 0)
	X_norm = (X - X_mean) / X_std

	return X_norm, X_mean, X_std


def read_file(file_name):
	'''
	Input filename and read dataset into ndarray.\n
	Output dataset ndarray.
	'''
	
	# Define location of dataset.
	current_directory = os.path.dirname(__file__)
	filepath = os.path.join(current_directory, '..', '..', 'data', file_name)

	file_ext = file_name.split('.')[-1]

	# Read data into ndarray.
	try:
		if file_ext == 'npy':
			# Numpy
				data = np.load(filepath)
		else:
			# Pandas
			data = pd.read_csv(filepath)
	except:
		sys.exit(filepath + ' not found.')


	return data