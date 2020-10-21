# project_3.py
#   Read, normalize, and visualize datasets.
# Devon Gardner
# 09/08/2020
#   python project_3.py -i <input_file> -x <x_header> -y <y_header> or py project_3.py -i <input_file> -x <x_header> -y <y_header>

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os, sys, getopt

def scatter(X, Y, title):
    '''Scatter plot two varaibles. X is the explanatory
    variable and Y is the response variable.'''
    plt.figure()
    plt.scatter(X, Y)
    plt.title(title)
    plt.xlabel(X.columns[0])
    plt.ylabel(Y.columns[0])


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


def regression(degree, X, Y, title):
    '''Create regression model based on given variables and degree (pandas dataframes)'''
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title + ' Single Model of D={0}'.format(degree))
    ax[0].plot(X, Y, 'ob')

    # Initialize A with ones.
    H = np.ones(X.shape)
    A = H

    # hstack X's for each degree
    for d in range(degree):
        A = np.hstack((X**(d + 1), A))

    # Calculate weights and predict y values
    W = np.linalg.inv(A.T @ A) @ A.T @ Y
    Y_pred = A @ W

    # Same as above but for synthetic values for smooth reg line
    X_synth = np.linspace(X.min(), X.max(), Y_pred.shape[0])
    A_synth = H
    for d in range(degree):
        A_synth = np.hstack((X_synth**(d + 1), A_synth))
    Y_synth = A_synth @ W

    # Plot smooth regression line
    ax[0].plot( X_synth, Y_synth, '-r', label="D={0}".format(degree))

    # Calcute regression stats
    R, RSq, MSE = stats(Y, Y_pred, degree)
    print('D: {0}, RSq: {1}, MSE: {2}'.format(degree, RSq.values, MSE.values))

    # Label everything
    ax[0].set_xlabel(X.columns[0])
    ax[0].set_ylabel(Y.columns[0])
    ax[0].set_title('W = {0}.T, R^2 = {1}'.format(W.values.tolist(), RSq.values))
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(X, np.abs(R), 'or')
    ax[1].set_xlabel(X.columns[0])
    ax[1].set_ylabel('|y - y*|')
    ax[1].set_title('MSE = {0}'.format(MSE.values))
    ax[1].grid()


def heatmap(cov, title, color_bar_label):
    '''Generate heatmap of given data'''
    fig, ax = plt.subplots()
    
    ax = sns.heatmap(cov, cmap = 'viridis', annot = True, fmt='.3f', square = True, cbar_kws = {'label': color_bar_label})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_title(title)


def pair_plot(data, vars, title, hue=None):
    '''Pair plot given dataset.'''
    sns.set(style="ticks", color_codes=True)
    pairplot = sns.pairplot(data=data, vars=vars, hue=hue)
    pairplot.fig.suptitle(title)


def norm_range(data):
    '''
    Normalize data by range.
    \nINPUTS: data (matrix)
    \nOUTPUTS: data normalized by range (matrix)
    '''
    # Prepare data for transformations by adding homogeneous 
    ones = np.ones((data.shape[0], 1))
    A = np.hstack((data, ones))

    # Compute some stats
    mins = np.min(A, axis=0).reshape((1, A.shape[1]))
    maxs = np.max(A, axis=0).reshape((1 ,A.shape[1]))
    ranges = (maxs - mins).reshape((1, A.shape[1]))

    # Translation matrix
    T = np.eye(A.shape[1])
    T[0:-1,-1] = -mins[0,0:-1]

    # Scale matrix
    S = np.eye(A.shape[1])
    S[0:-1,0:-1] = S[0:-1,0:-1] * (1/ranges[0,0:-1])

    # Normalization matrix
    N = S @ T

    # Normalize the dataset.
    A_norm = (N @ A.T).T

    # Remove homogeneous column.
    A_norm = np.delete(A_norm, -1, 1)

    return A_norm


def norm_z(data):
    '''
    Normalize data by standard deviation.
    \nINPUTS: data (matrix)
    \nOUTPUTS: data normalized by standard deviation (matrix)
    '''
    # Prepare data for transformations by adding homogeneous 
    ones = np.ones((data.shape[0], 1))
    A = np.hstack((data, ones))

    # Compute some stats
    means = np.mean(A, axis=0).reshape((1, A.shape[1]))
    stds = np.std(A, axis=0).reshape((1, A.shape[1]))

    # Translation matrix
    T = np.eye(A.shape[1])
    T[0:-1,-1] = -means[0,0:-1]

    # Scale matrix
    S = np.eye(A.shape[1])
    S[0:-1,0:-1] = S[0:-1,0:-1] * (1/stds[0,0:-1])

    # Normalization matrix
    N = S @ T

    # Normalize the dataset.
    A_norm = (N @ A.T).T

    # Remove homogeneous column.
    A_norm = np.delete(A_norm, -1, 1)

    return A_norm


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


def visualize(file_name, x_col, y_col):
    # Read in datasets.
    data = read_file(file_name)

    # Remove non numeric data.
    data_numeric = data.select_dtypes(exclude=object)
    data_numeric = data_numeric.dropna()
    data_numeric.reset_index(drop=True)

    # Normalize numeric data by range and restore dataframe metadata.
    data_norm_range = pd.DataFrame(norm_range(data_numeric.to_numpy()))
    data_norm_range.columns = data_numeric.columns

    # Normalize numeric data by standard deviation and restore dataframe metadata.
    data_norm_z = pd.DataFrame(norm_z(data_numeric.to_numpy()))
    data_norm_z.columns = data_numeric.columns

    # Pull user defined columns from DataFrame
    # Choose indexing method based on input type (str/int)
    if type(x_col) is str:
        X = pd.DataFrame(data[x_col])
        X_norm_z = pd.DataFrame(data_norm_z[x_col])
    else:
        X = pd.DataFrame(data[data.columns[x_col]])
        X_norm_z = pd.DataFrame(data_norm_z[data_norm_z.columns[x_col]])
    
    if type(y_col) is str:
        Y = pd.DataFrame(data[y_col])
        Y_norm_z = pd.DataFrame(data_norm_z[y_col])
    else:
        Y = pd.DataFrame(data[data.columns[y_col]])
        Y_norm_z = pd.DataFrame(data_norm_z[data_norm_z.columns[y_col]])

    # Visualize data.
    heatmap(data.cov(), file_name + ' Covariance', 'Covariance')
    heatmap(data_norm_range.cov(), file_name + ' Norm Range Covariance', 'Covariance')
    heatmap(data_norm_z.cov(), file_name + ' Norm Z Covariance', 'Covariance')

    scatter(X, Y, file_name)

    regression(1, X, Y, file_name)
    regression(2, X, Y, file_name)
    regression(3, X, Y, file_name)
    regression(4, X, Y, file_name)
    regression(5, X, Y, file_name)
    regression(6, X, Y, file_name)
    regression(7, X, Y, file_name)
    regression(8, X, Y, file_name)
    regression(9, X, Y, file_name)
    regression(10, X, Y, file_name)


def main(argv):
    input_file = ''
    x_col = ''
    y_col = ''
    try:
        opts, args = getopt.getopt(argv, "hi:x:y:")
    except getopt.GetoptError:
        print('usage: project_3.py -i <input_file> -x <x_header or x_col> -y <y_header or y_col>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: project_3.py -i <input_file> -x <x_header or x_col> -y <y_header or x_col>')
            sys.exit()
        elif opt == '-i':
            input_file = arg
        elif opt == '-x':
            if arg.isdigit() == False:
                x_col = arg
            else:
                x_col = int(arg)
        elif opt == '-y':
            if arg.isdigit() == False:
                y_col = arg
            else:
                y_col = int(arg)

    visualize(input_file, x_col, y_col)


if __name__=="__main__":
    main(sys.argv[1:])
    plt.show()