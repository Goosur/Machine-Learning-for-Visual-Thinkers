# optdigits.py -- demonstrate kmeans clustering
# 
# Devon Gardner
# Machine Learning for Visual Thinkers
# 10/20/2020
#
# to run in terminal: `py optdigits.py`
import ml

def clustering(filename, class_col):
	"""Cluster optdigits dataset using kmeans.

	Args:
		filename (str): filename
		class_col (int): Class column
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
		X_input = X.iloc[:, keepers]
	else:
		X_input = X

	# Center data on mean and perform PCA 
	X_centered, X_mean = ml.center(X_input)
	Y, P, e_scaled = ml.pca_cov( X_centered )
	headers = Y.columns

	# Grab sample numbers from dataset using boolean indexing
	samples = []
	nums = [0, 1, 2, 3, 4]
	for n in nums:
		samples.append(Y.loc[X.iloc[:, class_col] == n, :])

	# Concat and shuffle samples
	Y_sample = ml.pd.concat(samples, axis=0).reset_index(drop=True)
	Y_sample = Y_sample.sample(frac=1).reset_index(drop=True)

	# Numpify data for clustering because pandas is weird
	Y_sample = Y_sample.to_numpy()

	# Perform kmeans clustering.
	k = len(nums)
	clusters, means = ml.kmeans(Y_sample, k, headers)

	# Display kmeans clusters.
	ax = ml.vis_clusters(Y_sample, clusters, means, headers)

	# Display actual clusters.
	fig, ax = ml.plt.subplots()
	ax.set_title('Actual clusters')
	for n in nums:
		num = X.iloc[:, class_col] == n
		ax = ml.sns.scatterplot(Y.loc[num, Y.columns[0]], Y.loc[num, Y.columns[1]], label=n)


if __name__ == "__main__":
	clustering('optdigits.tra', -1)
	ml.plt.show()