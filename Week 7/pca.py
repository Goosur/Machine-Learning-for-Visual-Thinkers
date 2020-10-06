# project_2.py
#   Read, normalize, and visualize datasets.
# Devon Gardner
# 09/08/2020

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import project_2 as p2

def read_from_file(file_name):
    '''
    Input filename and read dataset into ndarray.\n
    Output dataset ndarray.
    '''
    
    # Define location of dataset.
    current_directory = os.path.dirname(__file__)
    filepath = os.path.join(current_directory, "..", "data", file_name)

    # Read data into ndarray.
    data = pd.read_csv(filepath)

    return data


def visualize(data, vars, title, hue=None):
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

    print("Minimum Value:\n", data.min(), sep='')
    print("\nMaximum Value:\n", data.max(), sep='')
    print("\nMedian:\n", data.median(), sep='')
    print("\nMean:\n", data.mean(), sep='')
    print("\nStandard Devation:\n", data.std(), sep='')
    print("\nVariance:\n", data.var(), sep='')
    print("\nCovariance:")
    print( data.cov() )


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
    iris_file = "iris.data"
    # cars_file = "Cars2015.csv"

    # Read in datasets.
    iris = read_from_file(iris_file)
    # cars = read_from_file(cars_file)

    # Remove non numeric data.
    iris_numeric = iris.select_dtypes(exclude=object)

    # Normalize numeric data by range and restore dataframe metadata.
    iris_norm_range = pd.DataFrame(norm_range(iris_numeric.to_numpy()))
    iris_norm_range.columns = iris_numeric.columns

    # Normalize numeric data by standard deviation and restore dataframe metadata.
    iris_norm_z = pd.DataFrame(norm_z(iris_numeric.to_numpy()))
    iris_norm_z.columns = iris_numeric.columns

    # Compute stats on numeric data and normalized numeric data.
    print("Iris numeric stats:\n")
    compute(iris_numeric)
    print("\nIris numeric normalized by range stats:\n")
    compute(iris_norm_range)
    print("\nIris numeric normalized by standard deviation stats:\n")
    compute(iris_norm_z)

    iris_norm_range = pd.concat([iris_norm_range, iris['Species']], axis = 1)
    iris_norm_z = pd.concat([iris_norm_z, iris['Species']], axis = 1)
    
    # Visualize data.
    visualize(data=iris, vars=iris_numeric.columns, title='Iris', hue=iris.columns[-1])
    visualize(data=iris_norm_range, vars=iris_numeric.columns, title='Iris', hue=iris.columns[-1])
    visualize(data=iris_norm_z, vars=iris_numeric.columns, title='Iris', hue=iris.columns[-1])


if __name__=="__main__":
    main()
    plt.show()