# project_4.py -- demonstrate PCA
# 
# Devon Gardner
# Machine Learning for Visual Thinkers
# 10/20/2020
#
# to run in terminal: project_4.py -i <input_file> -c <class_col> -s <sample_row>

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


def pca_svd( X ):
	"""Perform Principal Components Analysis (PCA) using the covariance matrix to identify principal components 
	(eigenvectors) and their scaled eigenvalues (which measure how much information each PC represents).
	
	INPUT:
	X -- (n,m) ndarray representing the dataset (observations), assuming one datum per row and one column per feature. 
			Must already be centered, so that the mean is at zero. Usually Z-score normalized. 
	
	OUTPUT:
	Y -- (n,m) ndarray representing rotated dataset (Y), 
	P -- (m,m) ndarray representing principal components (columns of P), a.k.a. eigenvectors
	e_scaled -- (m,) ndarray of scaled eigenvalues, which measure info retained along each corresponding PC """
	
	# Calculate principal components and eigenvalues using SVD
	( U, W, Vt ) = np.linalg.svd(X)
	e = W**2
	P = Vt.T

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


def reconstruct(X_input, X_mean, X_std, Y, P, e_scaled, dimensions, x_col = 0, y_col = 1):
	"""Reconstruct data from principle components.
	
	INPUT:
	X_input -- (n,m) dataframe representing the dataset (observations), assuming one datum per row and one column per feature. 
	X_mean -- (m,) series representing the means of the dataset by feature.
	X_std -- (m,) series representing the standard deviations o fthe dataset by feature.
	Y -- (n,m) dataframe representing the dataset after rotating onto principal components.
	P -- (m,m) dataframe representing the principal components calculated from the dataset.
	e_scaled -- (m,) series representing the data retention of the principal compoenents.

	OUTPUT:
	None"""
	# Reconstruction degrees information retention (~25%, ~50%, ~75%, and ~100%).
	for d in dimensions:
		# Reconstruct 
		Y_proj = Y.iloc[:,0:(d + 1)]
		X_rec = (Y_proj @ P.iloc[:,0:(d + 1)].T) * X_std + X_mean
		X_rec.columns = X_input.columns

		# Cumulate percentage information retained
		data_retained = e_scaled[range(d + 1)].sum() * 100

		plt.figure()
		plt.title(f'Raw vs. Reconstructed D = {d + 1}')
		sns.scatterplot(data = X_input, x = X_input.iloc[:, x_col], y = X_input.iloc[:, y_col], alpha = 0.5, color = 'k', label = 'Raw Data (100%)')
		sns.scatterplot(data = X_rec, x = X_rec.iloc[:, x_col], y = X_rec.iloc[:, y_col], alpha = 0.5, color = 'r', label = f'Reconstructed Data ({data_retained: .2f}%)')


def pca_analysis(filename, sample, class_col):
	''' Apply PCA to the specified dataset.'''

	X = read_file( filename )

	# Remove the class label from the dataset so that it doesn't prevent us from training a classifier in the future
	if class_col != None:
		try:
			classifier = pd.DataFrame(X.iloc[:, class_col])
		except:
			sys.exit('Class column out of range.')
		m = X.shape[1]
		keepers = list(range(m))
		keepers.pop( class_col )

	# Determine whether sample is present
	X_input = X.iloc[:, keepers]

	# # Visualize raw data
	# plt.figure()
	# sns.scatterplot(data = X, x = X_input['Petal Length (cm)'], y = X_input['Petal Width (cm)'], color = 'k', alpha = 0.5).set(title = filename + ' raw')

	# Normalize features by Z-score (so that features' units don't dominate PCs), and apply PCA
	X_norm, X_mean, X_std = z_norm(X_input)
	Y, P, e_scaled = pca_svd( X_norm )

	# # Visualize 2D PC data
	# plt.figure()
	# sns.scatterplot(data = Y, x = Y.iloc[:, 0], y = Y.iloc[:, 1], alpha=0.5, color = 'k').set(title = 'PC 2D Projection')

	# # Visualize PCs with heatmap and cree plot
	# info_retention = scree_plot( e_scaled )
	# pc_heatmap( P, info_retention )

	# # Reconstruct data
	# iris = [0, 1, 2, 3]
	# optdigits = []
	# lfwcrop = []
	# reconstruct(X_input, X_mean, X_std, Y, P, e_scaled, iris, 2, 3)


def main(argv):
	input_file = ''
	class_col = None
	sample = None
	try:
		opts, args = getopt.getopt(argv, "hi:c:s:")
	except getopt.GetoptError:
		print('usage: project_4.py -i <input_file> -c <class_col> -s <sample_row>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('usage: project_4.py -i <input_file> -c <class_col> -s <sample_row>')
			sys.exit()
		elif opt == '-i':
			input_file = arg
		elif opt == '-c':
			try:
				class_col = int(arg)
			except:
				sys.exit('Please enter a digit for class column.')
		elif opt == '-s':
			try:
				sample = int(arg)
			except:
				sys.exit('Please enter a digit for sample row.')			

	pca_analysis(input_file, sample, class_col)


if __name__=="__main__":
	main(sys.argv[1:])
	plt.show()