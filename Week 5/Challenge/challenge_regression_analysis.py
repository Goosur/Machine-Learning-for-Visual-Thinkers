# challenge_regression_analysis.py
#   Read and visualize datasets.
# Devon Gardner
# 09/08/2020

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

def read_from_file(file_name):
    '''
    Input filename and read dataset into ndarray.\n
    Output dataset ndarray.
    '''
    
    # Define location of dataset.
    current_directory = os.path.dirname(__file__)
    filepath = os.path.join(current_directory, "Data", file_name)

    # Read data into ndarray.
    data = pd.read_csv(filepath)

    return data

def visualize(data, vars, x1, y1, x2, title, hue=None):
    '''
    Plot the following types of visualizations: pair plot, joint plot, bar plot, and box plot.

    INPUTS\n
    data: ndarray\n
    vars: list, list of variables to be used in scatterplot\n
    x1: specific data variable x axis\n
    y2: specific data variable y axis\n
    x2: specific data variable x axis\n
    title: suptitle\n
    hue: data variable used for color coding visuals.\n

    OUTPUT\n
    None
    '''
    sns.set(style="ticks", color_codes=True)
    pairplot = sns.pairplot(data=data, vars=vars, hue=hue)
    pairplot.fig.suptitle(title)

def compute(data):
    '''
    Compute stats on inputted numeric data.\n 
    Stats to be computed: minimum value, maximum value, median, mean, standard devation, variance, covariance.
    '''

    print("\nMinimum Value:\n", data.min(), sep='')
    print("\nMaximum Value:\n", data.max(), sep='')
    print("\nMedian:\n", data.median(), sep='')
    print("\nMean:\n", data.mean(), sep='')
    print("\nStandard Devation:\n", data.std(), sep='')
    print("\nVariance:\n", data.var(), sep='')
    print("\nCovariance:")
    print( data.cov() )

def regression(X, Y):
    '''
    Calculate and plot simple regression of pandas dataframe. 
    \nInputs: Explanatory variable dataframe (X), response variable dataframe(Y)
    \nOutputs: None (plots data and regression line)
    '''
    # Homogeneous coordinate stacked on X

    H = np.ones(X.shape)
    A = np.hstack((X**2, X, H))

    # Simple linear regression
    W = np.linalg.lstsq(A, Y, rcond=None)[0]

    # Predict output based on W
    Y_pred = A @ W

    # Smooth regression line
    X_synth = np.linspace(np.min(X), np.max(X), Y_pred.shape[0])
    A_synth = np.hstack((X_synth**2, X_synth, H))
    Y_synth = A_synth @ W

    # Visualization
    plt.figure()
    plt.plot(X, Y, "ok", label=Y.columns[0])
    plt.plot(X_synth, Y_synth, '-r', linewidth=3, label=Y.columns[0] + ' Pred')
    plt.xlabel(X.columns[0])
    plt.ylabel(Y.columns[0])
    plt.grid()
    plt.legend()

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

def main():
    challenger_file = 'o-ring-erosion-or-blowby.data'

    # Read in datasets.
    challenger = read_from_file(challenger_file)

    # Print datasets and headers.
    # print('\nCHALLENGER HEADERS')
    # print(challenger.columns)
    # print('\nCHALLENGER DATA')
    # print(challenger)

    # Clean useless data.
    challenger = challenger.drop('O-Rings at Risk', axis=1)

    # Convert pandas dataframe to numpy for normalization then restore dataframe metadata.
    challenger_norm_range = pd.DataFrame(norm_range(challenger.to_numpy()))
    challenger_norm_range.columns = challenger.columns
    challenger_norm_z = pd.DataFrame(norm_z(challenger.to_numpy()))
    challenger_norm_z.columns = challenger.columns

    # Compute stats on numeric data.
    # compute(challenger)
    # compute(challenger_norm_range)
    # compute(challenger_norm_z)

    # Convert columns to one dimensional matrices.
    o_ring_risk = pd.DataFrame(challenger.iloc[:,0])
    temperature = pd.DataFrame(challenger.iloc[:,1])
    pressure = pd.DataFrame(challenger.iloc[:,2])

    # # Compute and visualize simple regression line.
    regression(temperature, o_ring_risk)
    # regression(pressure, o_ring_risk)


if __name__=="__main__":
    main()
    plt.show()