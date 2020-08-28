# mathematical_models.py
#   Visualize equations by generating sequences of values
#   within loops, and plotting them with matplotlib.
# Devon Gardner
# 08/24/2020

import matplotlib.pyplot as plt
import numpy as np
import math
import random

def main():

    # Modeling a straight line
    m = 0.667
    b = random.randint( -10, 10 )
    plt.figure( figsize=( 8, 4 ) )
    plt.xlabel( "x coordinate (units)" )
    plt.ylabel( "y coordinate (units)" )
    plt.title( "y = mx + b" )
    plt.grid()

    # Plotting each point individually
    for x in range( 10 ):
        y = m * x + b
        plt.plot( x, y, 'kd' )

    # Plotting all the points at once
    x_list = []
    y_list = []
    for x in range(10):
        y = m * x + b
        x_list.append( x )
        y_list.append( y )
    plt.plot( x_list, y_list, '--m', linewidth=4 )

    # Plotting a cubic polynomial
    x_list = []
    y_list = []
    a = random.random()
    b = random.randint( 0, 10 )
    c = random.gauss( 5, 2 )
    d = -3
    for x in range (10):
        y = a*x**3 + b*x**2 + c*x + d
        x_list.append( x )
        y_list.append( y )
    plt.plot ( x_list, y_list, "-c", linewidth=3 )

    # Plotting a sinusoid
    x_list = []
    y_list = []
    a = 2.5
    for x in range (10):
        y = a*np.sin( x )
        x_list.append( x )
        y_list.append( y )
    plt.plot ( x_list, y_list, "--r", linewidth=2 )

    # Plotting an arctangent
    x_list = []
    y_list = []
    for x in range (10):
        y = math.atan2( x, x-2 )
        x_list.append( x )
        y_list.append( y )
    plt.plot ( x_list, y_list, "-g", linewidth=1 )

    # NumPy preview:
    nd = np.arange( 10 ).reshape( (2,5) )
    print ( "NumPy ndarray\n", nd )
    print( "element[1][2]:", nd[1][2] )
    nd_times_two = nd*2
    print( "times two:\n", nd_times_two )

    # Make the figure appear!
    plt.show()


# Only execute the contents of this file when run
# directly, not when imported.
if __name__=="__main__":
    main()
