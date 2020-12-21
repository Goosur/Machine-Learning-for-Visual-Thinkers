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
	
	# Calculate principal components and eigenvalues using COV
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

	# Read data into ndarray.
	try:
		data = pd.read_csv(filepath)
	except:
		sys.exit(filepath + ' not found.')

	return data