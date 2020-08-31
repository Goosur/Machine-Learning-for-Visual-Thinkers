import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

def main():
    '''Imports Iris dataset and visualizes it.'''

    # Get current directory and set path to data file.
    current_directory = os.path.dirname(__file__)
    filepath = os.path.join(current_directory, "..", "data", "iris.data")

    # Import data as a pandas dataframe.
    iris_panda = pd.read_table( filepath, delimiter=',' )

    # Generate ndarray from data and convert it to pandas dataframe.
    iris = np.genfromtxt( filepath, delimiter=',', dtype='unicode' )
    iris_frame = pd.DataFrame(iris, columns=iris[0])
    iris_frame = iris_frame[1:]

    # Style, plot, and show data.
    sns.set(style="ticks", color_codes=True)
    frame = sns.pairplot(iris_frame, hue="Species")
    panda = sns.pairplot(iris_panda, hue="Species")
    plt.show()

    # Calculate Stats
    mean = np.mean( iris_panda )
    std = np.std( iris_panda )
    var = np.var( iris_panda )
    # cov = np.cov( iris_panda )

    # Display Stats
    print(mean)
    print(std)
    print(var)
    # print(cov)

main()