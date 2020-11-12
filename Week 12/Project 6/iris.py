import ml

def iris(filename, class_col):
	X = ml.read_file(filename)
	X_headers = X.columns

	# Split training and test data
	X_shuffled = X.sample(frac=1).reset_index(drop=True)

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
	Y, P, e_scaled = ml.pca_cov(X_norm)
	Y_headers = Y.columns

	# Give 1 test point and all training points
	# Use knn, classifier performance, and handwriting regognition lecture slides.