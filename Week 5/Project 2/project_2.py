# project_2.py
#   Read, normalize, and visualize datasets.
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
    filepath = os.path.join(current_directory, "..", "data", file_name)

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
    jointplot = sns.jointplot(data=data, x=x1, y=y1, hue=hue, kind="kde")
    jointplot.fig.suptitle(title)
    fig, ax = plt.subplots(ncols=2)
    fig.set_size_inches(16,8)
    plt.suptitle(title)
    bar = sns.barplot(x=data[x2], y=data[y1], ax=ax[0])
    box = sns.boxplot(x=data[x2], y=data[y1], ax=ax[1])

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

def main():
    iris_file = "iris.data"
    cars_file = "Cars2015.csv"

    # Read in datasets.
    iris = read_from_file(iris_file)
    cars = read_from_file(cars_file)

    # Print datasets and headers.
    print('\nIRIS HEADERS')
    print(iris.columns)
    print('\nIRIS DATA')
    print(iris)
    print('\nCARS HEADERS')
    print(cars.columns)
    print('\nCARS DATA')
    print(cars)

    # Remove non numeric data.
    iris_numeric = iris.select_dtypes(exclude=object)
    cars_numeric = cars.select_dtypes(exclude=object)

    # Compute stats on numeric data.
    compute(iris_numeric)
    compute(cars_numeric)

    # Visualize data.
    visualize(data=iris, vars=iris_numeric.columns, x1=iris.columns[2], y1=iris.columns[3], x2=iris.columns[4], title='Iris', hue=iris.columns[-1])
    visualize(data=cars, vars=cars_numeric.columns, x1=cars.columns[16], y1=cars.columns[6], x2=cars.columns[2], title='Cars', hue=cars.columns[2])

if __name__=="__main__":
    main()
    plt.show()