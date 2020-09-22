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

def regression(data):
    X = data.iloc[:,1].reshape((23,1))
    print(X)
    print(X.shape)
    Y = data.iloc[:,0]
    print(Y)
    print(Y.shape)

    # Homogeneous coordinate stacked on X
    H = pd.Series(np.ones((X.shape)))
    print(H)
    print(H.shape)
    A = pd.concat([X, H], axis = 1)
    print(A)
    print(A.shape)
    # Simple linear regression
    W = np.linalg.lstsq(A, Y, rcond=None)[0]
    print(W.shape)

    # Predict output based on W
    Y_pred = A @ W

    # Visualization
    plt.figure()
    plt.plot(X, Y, "ok", label=Y.name)
    plt.plot(X, Y_pred, '-r', linewidth=3, label=Y.name + ' Pred')
    plt.xlabel(X.name)
    plt.ylabel(Y.name)
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
    challenger_distress_pressure = challenger[['O-Rings in Distress', 'Leak-Check Pressure']]
    challenger_distress_temp = challenger[['O-Rings in Distress', 'Launch Temp']]

    # Compute stats on numeric data.
    compute(challenger_filtered)

    # Visualize data.
    #visualize(data=challenger_filtered, vars=challenger_filtered.columns, x1=challenger_filtered.columns[0], y1=challenger_filtered.columns[0], x2=challenger_filtered.columns[0], title='Challenger')
    regression(challenger_distress_pressure)
    regression(challenger_distress_temp)

if __name__=="__main__":
    main()
    plt.show()