# pca_cov_template.py -- A starting point for Thursday's covariance-based PCA challenge, if you like.
#
# Caitrin Eaton
# Machine Learning for Visual Thinkers
# Fall 2020
# 
# to run in terminal: python pca_cov_template.py

import numpy as np
import matplotlib.pyplot as plt
import os


def z_norm( X ):
	''' Normalize the dataset X by Z-score: subtract the mean and divide by the standard deviation.

	INPUT:
	X -- (n,m) ndarray of raw data, assumed to contain 1 row per sample and 1 column per feature.

	OUTPUT:
	X_norm -- (n,m) ndarray of Z-score normalized data
	X_mean -- (m,) ndarray of the means of the features (columns) of the raw dataset X
	X_std -- (m,) ndarray of the standard deviations of the features (columns) of the raw dataset X
	'''

	X_mean = np.mean( X, axis=0 )
	X_std = np.mean( X, axis=0 )
	X_norm = ( X - X_mean ) / X_std

	return X_norm, X_mean, X_std


def scree_plot( eigenvals ):
	"""Visualize information retention per eigenvector.
	
	INPUT:	
	eigenvals -- (d,) ndarray of scaled eigenvalues.
	
	OUTPUT:
	info_retention -- (d,) ndarray of accumulated information retained by multiple eigenvectors.  """
			
	# Visaulize individual information retention per eigenvector (eigenvalues)
	fig, ax = plt.subplots( 2, 1 )
	ax[0].plot( eigenvals, '-o', linewidth=2, markersize=5, markerfacecolor="w" )
	ax[0].set_ylim([-0.1, 1.1])
	ax[0].set_title( "Information retained by individual PCs" )
	ax[0].grid( True )
	
	# Visualize accumulated information retained by multiple eigenvectors
	info_retention = np.cumsum( eigenvals )
	ax[1].plot( info_retention, '-o', linewidth=2, markersize=5, markerfacecolor="w" )
	ax[1].set_ylim([-0.1, 1.1])
	ax[1].set_title( "Cumulative information retained by all PCs" )
	ax[1].grid( True )
		
	return info_retention


def pc_heatmap( P, info_retention ):
	''' Visualize principal components (eigenvectors) as a heatmap. 
	
	INPUT:
	P -- (m,m) ndarray of principal components (eigenvectors)
	info_retention -- (m,) ndarray of accumulated scaled eigenvectors: the % info retained by all PCs
	
	OUTPUT: 
	None
	'''
	plt.figure()
	plt.title("PC Heatmap")
	plt.imshow(P)


def read_file( filename ):
	''' Read in the data from a file.

	INPUT:
	filename -- string representing the name of a file in the "../data/" directory

	OUTPUT:
	data -- (n,m) ndarray of data from the specified file, assuming 1 row per sample and 1 column per feature
	headers -- list of length m representing the name of each feature (column)
	'''
	
	# Windows is kind of a jerk about filepaths. My relative filepath didn't
	# work until I joined it with the current directory's absolute path.
	current_directory = os.path.dirname(__file__)
	filepath = os.path.join(current_directory, "data", filename)
	print( "\nfilepath:", filepath )

	# Read headers from the 1st row with plain vanilla Python file handling (without Numpy)
	in_file = open( filepath )
	headers = in_file.readline().split(",")
	in_file.close()

	# Read iris's data in, skipping the metadata in 1st row
	data = np.genfromtxt( filepath, delimiter=",", skip_header=1 )
	
	return data, headers


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
	
	# Pull principal components and eigenvalues from covariance matrix
	C = np.cov(X, rowvar = False)
	(e, P) = np.linalg.eig(C)

	# Sort principal components in order of descending eigenvalues
	order = np.argsort(e)[::-1]
	e = e[order]
	P = P[:, order]
	
	# Scale eigenvalues to calculate the percent info retained along each PC
	e_scaled = e / e.sum()
	
	# Rotate data onto the principal components
	Y = X @ P
	
	return (Y, P, e_scaled)


def pca_analysis( filename="iris_preproc.csv", class_col=-1 ):
	''' Apply PCA to the specified dataset.'''
	
	X, headers = read_file( filename )
	
	# Remove the class label from the dataset so that it doesn't prevent us from training a classifier in the future
	species = X[:, class_col]
	m = X.shape[1]
	keepers = list(range(m))
	keepers.pop( class_col )
	X_input = X[:, keepers]
	
	# Sanity check
	print( "\nOriginal headers:\n\n", headers, "\n" )
	print( "\nOriginal dataset:\n\n", X[:5,:], "\n" )
	print( "\nWithout class col:\n\n", X_input[:5,:], "\n" )

	# Visualize raw data
	plt.figure()
	plt.plot( X_input[:,0], X_input[:,1], 'ob', alpha=0.5, markeredgecolor='w' )		
	plt.title( "Raw data" )
	plt.tight_layout()
		
	# Normalize features by Z-score (so that features' units don't dominate PCs), and apply PCA
	X_norm, X_mean, X_std = z_norm( X_input )
	Y, P, e_scaled = pca_cov( X_norm )
	
	# Sanity check: Print PCs and eigenvalues in the terminal
	print( "Eigenvectors (each column is a PC): \n\n", P, "\n" )
	print("\nScaled eigenvalues: \t", e_scaled, "\n" )
	
	# Visualize PCs with heatmap and cree plot
	info_retention = scree_plot( e_scaled )
	pc_heatmap( P, info_retention )

	# Project data onto PCs and reconstruct
	Y_proj = Y[:,0:2]
	X_rec = (Y_proj @ P[:,0:2].T) * X_std + X_mean

	# Visualize 2D PC data
	plt.figure()
	plt.plot( Y[:,0], Y[:,1], 'ob', alpha=0.5, markeredgecolor='w' )		
	plt.title( "PC 2D Projection" )
	plt.tight_layout()


if __name__=="__main__":
	#pca_analysis( "iris_preproc.csv", class_col=4 )
	pca_analysis( "wine.data", class_col=1 )
	plt.show()