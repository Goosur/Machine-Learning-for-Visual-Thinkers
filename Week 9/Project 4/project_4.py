# project_4.py -- demonstrate PCA
# 
# Devon Gardner
# Machine Learning for Visual Thinkers
# 10/20/2020
#
# to run in terminal: python project_4.py -i <input_file> -c <class_col>

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os, sys, getopt

def pc_heatmap( P, info_retention ):
	''' Visualize principal components (eigenvectors) as a heatmap. 
	
	INPUT:
	P -- (m,m) ndarray of principal components (eigenvectors)
	info_retention -- (m,) ndarray of accumulated scaled eigenvectors: the % info retained by all PCs
	
	OUTPUT: 
	None
	'''
	fig, ax = plt.subplots()

	ax = sns.heatmap(P.abs(), cmap = 'bone', vmin = 0, vmax = 1, annot = True, fmt='.3f', square = True)
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
	
	# Pull principal components and eigenvalues from covariance matrix

	C = X.cov()
	(e, P) = np.linalg.eig(C)
	e_len = len(e)

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


def stats(Y, Y_pred, d):
	'''Calculate regression stats from given actual/predicted values and degree.'''
	R = Y - Y_pred
	mean_r = np.mean( R )

	RSS = np.sum(R**2)

	mean = np.mean(Y)
	SS = np.sum((Y - mean)**2)
	RSq = 1 - (RSS / SS)

	MSE = RSS / R.shape[0]

	return R, RSq, MSE


def heatmap(cov, title, color_bar_label):
	'''Generate heatmap of given data'''
	fig, ax = plt.subplots()
	
	ax = sns.heatmap(cov, cmap = 'viridis', annot = True, fmt='.3f', square = True, cbar_kws = {'label': color_bar_label})
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
	ax.set_title(title)


def z_norm(X):
	''' Normalize the dataset X by Z-score: subtract the mean and divide by the standard deviation.

	INPUT:
	X -- (n,m) ndarray of raw data, assumed to contain 1 row per sample and 1 column per feature.

	OUTPUT:
	X_norm -- (n,m) ndarray of Z-score normalized data
	X_mean -- (m,) ndarray of the means of the features (columns) of the raw dataset X
	X_std -- (m,) ndarray of the standard deviations of the features (columns) of the raw dataset X
	'''

	X_mean = X.mean(axis=0)
	X_std = X.std(axis=0)
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
	data = pd.read_csv(filepath)

	return data


def pca_analysis(filename="iris_preproc.csv", class_col=-1):
	''' Apply PCA to the specified dataset.'''
	
	X = read_file( filename )
		   
	# Remove the class label from the dataset so that it doesn't prevent us from training a classifier in the future
	classifier = pd.DataFrame(X[X.columns[class_col]])
	m = X.shape[1]
	keepers = list(range(m))
	keepers.pop( class_col )
	X_input = X[X.columns[keepers]]
	
	# # Sanity check
	# print( "\nOriginal headers:\n\n", X.columns, "\n" )
	# print( "\nOriginal dataset:\n\n", X, "\n" )
	# print( "\nWithout class col:\n\n", X_input, "\n" )

	# # Visualize raw data
	# plt.figure()
	# sns.scatterplot(data = X, x = X.columns[0], y = X.columns[1], hue = X.iloc[:, class_col].tolist(), palette = 'Dark2').set(title = filename + ' raw w/ classes')
		
	# Normalize features by Z-score (so that features' units don't dominate PCs), and apply PCA
	X_norm, X_mean, X_std = z_norm(X_input)
	Y, P, e_scaled = pca_cov( X_norm )

	# # Sanity check: Print PCs and eigenvalues in the terminal
	# print( "Eigenvectors (each column is a PC): \n", P, "\n", sep = '')
	# print("\nScaled eigenvalues: \n", e_scaled, "\n", sep = '')
	
	# # Visualize PCs with heatmap and cree plot
	# info_retention = scree_plot( e_scaled )
	# pc_heatmap( P, info_retention )

	# # Visualize 2D PC data
	# plt.figure()
	# sns.scatterplot(data = Y, x = Y.iloc[:, 0], y = Y.iloc[:, 1], alpha=0.5).set(title = 'PC 2D Projection')

	# Project data onto PCs and reconstruct
	d_max = len(Y.columns)
	for d in range(d_max + 1):
		Y_proj = Y.iloc[:,0:d]
		X_rec = (Y_proj @ P.iloc[:,0:d].T) * X_std + X_mean
		X_rec.columns = X_input.columns

		plt.figure()
		plt.title('Raw vs. Reconstructed D = {0}'.format(d))
		sns.scatterplot(data = X_input, x = X_input['petal length (cm)'], y = X_input['petal width (cm)'], alpha = 0.5, color = 'k', label = 'Raw Data')
		sns.scatterplot(data = X_rec, x = X_rec['petal length (cm)'], y = X_rec['petal width (cm)'], alpha = 0.5, color = 'r', label = 'Reconstructed Data')


def main(argv):
	input_file = ''
	class_col = -1
	try:
		opts, args = getopt.getopt(argv, "hi:c:")
	except getopt.GetoptError:
		print('usage: project_4.py -i <input_file> -c <class_col>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('usage: project_4.py -i <input_file> -c <class_col>')
			sys.exit()
		elif opt == '-i':
			input_file = arg
		elif opt == '-c':
			class_col = int(arg)

	pca_analysis(input_file, class_col)


if __name__=="__main__":
	main(sys.argv[1:])
	plt.show()