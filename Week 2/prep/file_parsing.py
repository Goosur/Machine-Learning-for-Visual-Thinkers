import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

iris_panda = pd.read_table('iris.data', delimiter=',')
iris = np.genfromtxt('iris.data', delimiter=',', dtype='unicode')
iris_frame = pd.DataFrame(iris, columns=iris[0])
iris_frame = iris_frame[1:]

sns.set(style="ticks", color_codes=True)
frame = sns.pairplot(iris_frame, hue="class")
panda = sns.pairplot(iris_panda, hue="class")
plt.show()