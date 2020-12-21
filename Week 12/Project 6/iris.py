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
		classes = X.iloc[:, class_col]
	else:
		X_input = X

	# # Visualize raw data
	# ml.plt.figure()
	# ml.sns.scatterplot(data = X, x = X_input['Petal Length (cm)'], y = X_input['Petal Width (cm)'], color = 'k', alpha = 0.5).set(title = filename + ' raw')


	X_tra = X_input.sample(frac=0.6, random_state=0)
	X_test = X_input.drop(X_tra.index)
	
	# Center data on mean and perform PCA 
	X_tra_norm, X_tra_mean, X_tra_std = ml.z_norm(X_tra)
	Y, P, e_scaled = ml.pca_cov(X_tra_norm)
	Y_headers = Y.columns

	# Reattach classes to training set
	Y_tra = Y.join(classes)
	
	# Normalize test set by training set
	X_test_norm = (X_test - X_tra_mean) / X_tra_std

	# Rotate test set onto training set PCA
	Y_test = X_test_norm.dot(P.to_numpy())
	Y_test.columns = P.columns

	# Perform KNN
	k = 30
	c_pred = ml.knn(Y_tra, Y_test, k)
	c_pred.rename('Species Predicted', inplace=True)
	c_actual = classes.iloc[c_pred.index]
	c_actual.rename('Species Actual', inplace=True)

	# Create confusion matrix
	classes_confusion = ml.pd.crosstab(c_actual, c_pred)
	
	# Calculate performance metrics for KNN
	# Number of FP/FN/TP/TN
	false_positive = classes_confusion.sum(axis=0) - ml.np.diag(classes_confusion)
	false_negative = classes_confusion.sum(axis=1) - ml.np.diag(classes_confusion)
	true_positive = ml.np.diag(classes_confusion)
	true_negative = classes_confusion.values.sum() - (false_positive + false_negative + true_positive)

	# Rate of TP/FP/Precision
	TP = true_positive / (true_positive + false_negative)
	TP.rename('True Positive Rate', inplace=True)
	FP = false_positive / (false_positive + true_negative)
	FP.rename('False Positive Rate', inplace=True)
	precision = true_positive / (true_positive + false_positive)
	precision.rename('Precision', inplace=True)
	print('Number of neighbors (k):', k)
	print(TP, end='\n\n')
	print(FP, end='\n\n')
	print(precision, end='\n\n')
	print(classes_confusion)

	# Visualize confusion matrix
	ml.plt.figure()
	ml.plt.title(f'Confusion Matrix; K={k}')
	ml.sns.heatmap(classes_confusion, cmap='bone', annot=True)


iris('iris.data', -1)
ml.plt.show()