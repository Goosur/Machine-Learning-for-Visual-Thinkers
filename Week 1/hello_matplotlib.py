# hello_matplotlib.py
#   Make sure Python3 and matplotlib are installed and working.
# Devon Gardner
# 08/24/2020

# Test Python 3 with a simple print statement
print("Hello, world!")

# Test matplotlib
import matplotlib.pyplot as plt

# plt.plot( [1, 2, 3, 4] )
# plt.plot( [1, 2], [2.5, 1.5], 'ro' )
# plt.ylabel( 'some numbers' )
# plt.show()

# Model a line: y = mx + b
m = 0.667
b = 1
plt.figure( figsize=( 8, 4 ) )
plt.xlabel( "x coordinate (units)" )
plt.ylabel( "y coordinate (units)" )
plt.title( "y = mx + b" )
plt.grid()
for x in range( 10 ):
    y = m * x + b
    plt.plot( x, y, 'kd' )
plt.show()