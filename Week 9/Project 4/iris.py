# project_4.py -- demonstrate PCA
# 
# Devon Gardner
# Machine Learning for Visual Thinkers
# 10/20/2020
#
# to run in terminal: project_4.py -i <input_file> -c <class_col> -s <sample_row>
import ml

def reconstruct(X_input, X_mean, X_std, Y, P, e_scaled, x_col = 0, y_col = 1, dimensions = [0, 1, 2, 3]):
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

		ml.plt.figure()
		ml.plt.title(f'Raw vs. Reconstructed D = {d + 1}')
		ml.sns.scatterplot(data = X_input, x = X_input.iloc[:, x_col], y = X_input.iloc[:, y_col], alpha = 0.5, color = 'k', label = 'Raw Data (100%)')
		ml.sns.scatterplot(data = X_rec, x = X_rec.iloc[:, x_col], y = X_rec.iloc[:, y_col], alpha = 0.5, color = 'r', label = f'Reconstructed Data ({data_retained: .2f}%)')


def pca(filename, class_col, sample):
	"""Perform PCA on given dataset.

	Args:
		filename (str): Filename with extension.
		sample (int): Sample row.
		class_col (int): Class label column.
	"""

	X = ml.read_file( filename )

	# Remove the class label from the dataset so that it doesn't prevent us from training a classifier in the future
	if class_col != None:
		try:
			classifier = ml.pd.DataFrame(X.iloc[:, class_col])
		except:
			ml.sys.exit('Class column out of range.')
		m = X.shape[1]
		keepers = list(range(m))
		keepers.pop( class_col )

	# Determine whether sample is present
	X_input = X.iloc[:, keepers]

	# # Visualize raw data
	ml.plt.figure()
	ml.sns.scatterplot(data = X, x = X_input['Petal Length (cm)'], y = X_input['Petal Width (cm)'], color = 'k', alpha = 0.5).set(title = filename + ' raw')

	# Normalize features by Z-score (so that features' units don't dominate PCs), and apply PCA
	X_norm, X_mean, X_std = ml.z_norm(X_input)
	Y, P, e_scaled = ml.pca_cov( X_norm )

	# Visualize 2D PC data
	ml.plt.figure()
	ml.sns.scatterplot(data = Y, x = Y.iloc[:, 0], y = Y.iloc[:, 1], alpha=0.5, color = 'k').set(title = 'PC 2D Projection')

	# Visualize PCs with heatmap and cree plot
	info_retention = ml.scree_plot( e_scaled )
	ml.pc_heatmap( P, info_retention )

	# Reconstruct data
	reconstruct(X_input, X_mean, X_std, Y, P, e_scaled, 2, 3)

	ml.plt.show()

pca('iris.data', -1, 0)