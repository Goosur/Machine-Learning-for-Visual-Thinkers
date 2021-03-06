# project_4.py -- demonstrate PCA
# 
# Devon Gardner
# Machine Learning for Visual Thinkers
# 10/20/2020
#
# to run in terminal: project_4.py -i <input_file> -c <class_col> -s <sample_row>
from numpy.core.fromnumeric import shape
import ml

def read_lfwcrop(filename, id_filename):
    current_directory = ml.os.path.dirname(__file__)
    faces_filepath = ml.os.path.join(current_directory, '..', '..', 'data', filename)
    lfw_faces = ml.np.load(faces_filepath)

    names_filepath = ml.os.path.join(current_directory, '..', '..', 'data', id_filename)
    lfw_names = ml.np.loadtxt(names_filepath, dtype=str, delimiter='\n')

    return lfw_faces, lfw_names


def reconstruct(X_input, X_mean, Y, P, e_scaled, sample):
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
	# Find dimensions (~25%, ~50%, ~75%, and ~100%).
	percentages = [0.25, 0.5, 0.75, 1]
	dimensions = []
	e_cumsum = e_scaled.cumsum()
	for p in percentages:
		dimensions.append(e_cumsum.searchsorted(p) - 1)

	# Reconstruction degrees information retention (~25%, ~50%, ~75%, and ~100%).
	for d, p in zip(dimensions, percentages):
		# Reconstruct 
		Y_proj = Y.iloc[:,0:(d + 1)]
		X_rec = (Y_proj @ P.iloc[:,0:(d + 1)].T) + X_mean
		X_rec.columns = X_input.columns

		# Cumulate percentage information retained
		data_retained = e_scaled[range(d + 1)].sum() * 100

		ml.plt.figure()
		ml.plt.title(f'Reconstructed D = {d + 1}, Info Retention: {p*100:.0f}%')
		X_rec_sample = ml.pd.DataFrame(X_rec.iloc[sample, :]).T.values.reshape((64, 64))
		ml.sns.heatmap(X_rec_sample, cmap='bone')


def pca(filename, ids, sample):
	"""Perform PCA on given dataset.

	Args:
		filename (str): Filename with extension.
		sample (int): Sample row.
		class_col (int): Class label column.
	"""

	X, X_ids = read_lfwcrop(filename, ids)
	n = X.shape[0]
	m = X.shape[1]*X.shape[2]
	X_input = ml.pd.DataFrame(X.reshape((n, m)))
	X_ids = ml.pd.Series(X_ids)

	# Visualize raw data
	ml.plt.figure()
	ml.plt.title(X_ids.iloc[sample])
	X_sample = ml.pd.DataFrame(X_input.iloc[sample, :]).T.values.reshape((X.shape[1], X.shape[2]))
	ml.sns.heatmap(X_sample, cmap='bone')

	# Normalize features by Z-score (so that features' units don't dominate PCs), and apply PCA
	X_centered, X_mean = ml.center(X_input)
	Y, P, e_scaled = ml.pca_cov( X_centered )

	# # Visualize 2D PC data
	# ml.plt.figure()
	# Y_sample = ml.pd.DataFrame(Y.iloc[sample, :]).T.values.reshape((X.shape[1], X.shape[2]))
	# ml.sns.heatmap(Y_sample, cmap='bone')

	# # Visualize PCs with heatmap and cree plot
	# info_retention = ml.scree_plot( e_scaled )
	# ml.pc_heatmap( P, info_retention )

	# Reconstruct data
	reconstruct(X_input, X_mean, Y, P, e_scaled, sample)

	ml.plt.show()

pca('lfwcrop.npy', 'lfwcrop_ids.txt', 0)