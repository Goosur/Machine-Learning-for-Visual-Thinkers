import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

current_directory = os.path.dirname(__file__)
filepath = os.path.join(current_directory, "..", "data", "iris.data")

iris_panda = pd.read_table( filepath, delimiter=',' )
iris = np.genfromtxt( filepath, delimiter=',', dtype='unicode' )
iris_frame = pd.DataFrame(iris, columns=iris[0])
iris_frame = iris_frame[1:]

sns.set(style="ticks", color_codes=True)
frame = sns.pairplot(iris_frame, hue="class")
panda = sns.pairplot(iris_panda, hue="class")
#plt.show()

print(iris_frame)
print(iris_panda)