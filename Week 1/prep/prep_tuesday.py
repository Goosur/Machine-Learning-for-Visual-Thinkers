# prep_tuesday.py
#   Loop based ballistic simulation.
# Devon Gardner
# 08/25/2020

import matplotlib.pyplot as plt
import numpy as np
import time

launch_angle = 60
v_init = 20
t = 0
g = -9.81
x = 0
y = 0
x_list = []
y_list = []

while y >= 0:
    t += 0.001
    x_list.append( x )
    y_list.append( y )
    x = v_init * np.cos( launch_angle * np.pi/180.0 ) * t
    y = v_init * np.sin( launch_angle * np.pi/180.0 ) * t + 0.5 * g * t**2

plt.xlabel( "Horizontal Position (m)" )
plt.ylabel( "Vertical Position (m)" )
plt.title( "Trajectory of a Ballistic Projectile" )
plt.grid()
plt.plot( x_list, y_list, 'kd' )
plt.show()