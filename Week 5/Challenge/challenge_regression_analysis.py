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
    # jointplot = sns.jointplot(data=data, x=x1, y=y1, hue=hue, kind="kde")
    # jointplot.fig.suptitle(title)
    # fig, ax = plt.subplots(ncols=2)
    # fig.set_size_inches(16,8)
    # plt.suptitle(title)
    # bar = sns.barplot(x=data[x2], y=data[y1], ax=ax[0])
    # box = sns.boxplot(x=data[x2], y=data[y1], ax=ax[1])

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

def regression(X, Y):
    '''
    Calculate and plot simple regression of pandas dataframe. 
    \nInputs: Explanatory variable dataframe (X), response variable dataframe(Y)
    \nOutputs: None (plots data and regression line)
    '''
    # Homogeneous coordinate stacked on X
    H = np.ones(X.shape)
    A = np.hstack((X, H))

    # Simple linear regression
    W = np.linalg.lstsq(A, Y, rcond=None)[0]

    # Predict output based on W
    Y_pred = A @ W

    # Visualization
    plt.figure()
    plt.plot(X, Y, "ok", label=Y.columns[0])
    plt.plot(X, Y_pred, '-r', linewidth=3, label=Y.columns[0] + ' Pred')
    plt.xlabel(X.columns[0])
    plt.ylabel(Y.columns[0])
    plt.grid()
    plt.legend()

def main():
    # Read in datasets.
    challenger = read_from_file('o-ring-erosion-or-blowby.data')

    # Print datasets and headers.
    print('\nCHALLENGER HEADERS')
    print(challenger.columns)
    print('\nCHALLENGER DATA')
    print(challenger)

    # Filter useless data.
    challenger_filtered = challenger.drop('O-Rings at Risk', axis = 1)

    # Compute stats on numeric data.
    compute(challenger)

    # Convert columns to one dimensional matrices.
    o_ring_risk = pd.DataFrame(challenger.iloc[:,1])
    temperature = pd.DataFrame(challenger.iloc[:,2])
    pressure = pd.DataFrame(challenger.iloc[:,3])

    # Visualize data.
    #visualize(data=challenger_filtered, vars=challenger_filtered.columns, x1=challenger_filtered.columns[0], y1=challenger_filtered.columns[0], x2=challenger_filtered.columns[0], title='Challenger')
    regression(temperature, o_ring_risk)
    regression(pressure, o_ring_risk)

if __name__=="__main__":
    main()
    plt.show()