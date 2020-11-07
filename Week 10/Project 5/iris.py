# iris.py -- demonstrate kmeans clustering
# 
# Devon Gardner
# Machine Learning for Visual Thinkers
# 10/20/2020
#
# to run in terminal: `py iris.py`
from matplotlib.pyplot import cla
import ml

def vis_clusters_actual(X, X_input, classes, class_col):
	fig, ax = ml.plt.subplots()
	ax.set_title('Actual clusters')
	for c in classes:
		clss = X.iloc[:, class_col] == c
		ax = ml.sns.scatterplot(X_input.loc[clss, X_input.columns[0]], X_input.loc[clss, X_input.columns[1]], label=c)


def sample(X, X_input, nums, class_col):
	# Grab sample numbers from dataset using boolean indexing
	samples = []
	for n in nums:
		samples.append(X_input.loc[X.iloc[:, class_col] == n, :])

	# Concat and shuffle raw samples
	X_sample = ml.pd.concat(samples, axis=0).reset_index(drop=True)
	X_sample = X_sample.sample(frac=1).reset_index(drop=True)

	return X_sample


def clustering(filename, class_col):
	"""Cluster optdigits dataset using kmeans.

	Args:
		filename (str): filename
		class_col (int): Class column
	"""
	X = ml.read_file( filename )
	X_headers = X.columns

	# Remove the class label from the dataset so that it doesn't prevent us from training a classifier in the future
	if class_col != None:
		try:
			classifier = ml.pd.DataFrame(X.iloc[:, class_col])
		except:
			ml.sys.exit('Class column out of range.')
		m = X.shape[1]
		keepers = list(range(m))
		keepers.pop( class_col )
		X_input = X.iloc[:, keepers]
	else:
		X_input = X

	# Center data on mean and perform PCA 
	X_norm, X_mean, X_std = ml.z_norm(X_input)
	Y, P, e_scaled = ml.pca_cov( X_norm )
	Y_headers = Y.columns

	# Sample and shuffle raw and pca data
	classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
	X_sample = sample(X, X_input, classes, class_col)
	Y_sample = sample(X, Y, classes, class_col)
	print(X_sample.shape)
	print(Y_sample.shape)
	# Numpify data for clustering because pandas is weird
	X_sample = X_sample.to_numpy()
	Y_sample = Y_sample.to_numpy()

	# Perform kmeans clustering on raw data.
	k = 5
	X_clusters, X_means = ml.kmeans(X_sample, k, X_headers)

	# Perform kmeans clustering on pca data.
	Y_clusters, Y_means = ml.kmeans(Y_sample, k, Y_headers)

	# Display kmeans clusters.
	ax = ml.vis_clusters(X_sample, X_clusters, X_means, X_headers)
	ax = ml.vis_clusters(Y_sample, Y_clusters, Y_means, Y_headers)

	# Display actual clusters.
	vis_clusters_actual(X, X_input, classes, class_col)
	vis_clusters_actual(X, Y, classes, class_col)


if __name__ == "__main__":
	clustering('iris.data', -1)
	ml.plt.show()