import matplotlib.pyplot as plt
import numpy as np
import time

def non_vectorized():
    launch_angle = 60
    v_init = 20
    t = 0
    g = -9.81
    x = 0
    y = 0
    x_list = []
    y_list = []

    while y >= 0:
        t += 0.003534
        x_list.append( x )
        y_list.append( y )
        x = v_init * np.cos( launch_angle * np.pi/180.0 ) * t
        y = v_init * np.sin( launch_angle * np.pi/180.0 ) * t + 0.5 * g * t**2

    plt.xlabel( "Horizontal Position (m)" )
    plt.ylabel( "Vertical Position (m)" )
    plt.title( "Trajectory of a Ballistic Projectile" )
    plt.grid()
    plt.plot( x_list, y_list, 'kd' )
    #plt.show()
    return len(x_list)

def vectorized():
    n = 1000
    launch_angle = 60
    v_init = 20
    g = -9.81
    t_range = 0, ( -v_init * np.sin( launch_angle * np.pi/180.0 ) ) / ( 0.5 * g )

    t = np.linspace( *t_range, n )
    x = v_init * np.cos( launch_angle * np.pi/180.0 ) * t
    y = v_init * np.sin( launch_angle * np.pi/180.0 ) * t + 0.5 * g * t**2

    plt.xlabel( "Horizontal Position (m)" )
    plt.ylabel( "Vertical Position (m)" )
    plt.title( "Trajectory of a Ballistic Projectile" )
    plt.grid()
    plt.plot( x, y, 'k.' )
    #plt.show()
    return x.shape

def runtime(function, runs):
    times = []

    # for every run calculate runtime and append to list of times
    for run in range(runs):
        time_start = time.time()
        function()
        times.append( time.time() - time_start )
        
    # calculate and print average runtime from list of times and number of runs
    time_avg = sum(times) / runs
    print("Running", function.__name__, runs, "times takes on average", time_avg*10**6, "micro seconds")

runs = 1000

print("x loop length:", non_vectorized())
print("x vector shape:", vectorized())

runtime(non_vectorized, runs)
runtime(vectorized, runs)