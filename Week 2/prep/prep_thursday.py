# prep_thursday.py
#   Test np.mean() with odd dataset.
# Devon Gardner
# 09/03/2020

import matplotlib.pyplot as plt
import numpy as np
import os

current_directory = os.path.dirname(__file__)
filepath = os.path.join(current_directory, "..", "data", "Goldie2019.csv")

Goldie2019 = np.isfinite(np.genfromtxt( filepath, delimiter=',' ))

print(Goldie2019)
print(np.mean(Goldie2019, axis=0))