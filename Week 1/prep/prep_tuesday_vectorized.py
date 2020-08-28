# prep_tuesday.py
#   Loop based ballistic simulation.
# Devon Gardner
# 08/25/2020

import matplotlib.pyplot as plt
import numpy as np
import time

launch_angle = 60
v_init = 20
g = -9.81
# t_init = 0
# t_final = ( -v_init * np.sin( launch_angle * np.pi/180.0 ) ) / ( 0.5 * g )
t_range = 0, ( -v_init * np.sin( launch_angle * np.pi/180.0 ) ) / ( 0.5 * g )

t = np.linspace( *t_range )
x = v_init * np.cos( launch_angle * np.pi/180.0 ) * t
y = v_init * np.sin( launch_angle * np.pi/180.0 ) * t + 0.5 * g * t**2

plt.xlabel( "Horizontal Position (m)" )
plt.ylabel( "Vertical Position (m)" )
plt.title( "Trajectory of a Ballistic Projectile" )
plt.grid()
plt.plot( x, y, 'k.' )
plt.show()