import numpy as np
import matplotlib.pyplot as plt
import os

# Since we don't care too much about decimal places, today, let's make the
# output a little more human friendly. Heads up: This isn't always a good idea!
np.set_printoptions(precision=1, suppress=True)

def read_Goldie_from_file( ):
    # 1. Read in the data file.
    #       -- This os.path.join() assumes that Goldie2019.csv is saved at the relative path "../data"
    #       -- You may have to alter this line to match your directory structure.
    current_directory = os.path.dirname(__file__)
    filepath = os.path.join(current_directory, "data", "Goldie2019.csv")
    print( "\nfilepath:", filepath )

    # Read Goldie's headers and units from the 3rd and 4th rows of the file
    goldie_file = open( filepath )
    _ = goldie_file.readline().split(",")
    _ = goldie_file.readline().split(",")
    feature_names = goldie_file.readline().split(",")
    feature_units = goldie_file.readline().split(",")
    goldie_file.close()
    print("Column 4 metadata:", feature_names[4] + "(" + feature_units[4] + ")")
    print("Column 10 metadata:", feature_names[10] + "(" + feature_units[10] + ")")

    # Read the weather buoy data from Goldie2019.csv into a Numpy ndarray.
    # Watch out for the top 4 rows of metadata using the optional skip_header parameter.
    goldie_data = np.genfromtxt( filepath, delimiter=",", skip_header=4)
    print( "Goldie data matrix shape:", goldie_data.shape )

    return goldie_data, feature_names, feature_units


def explore( data, x_name="Air Temp (C)", y_name="1m Water Temp (C)", title="" ):
    ''' Print some stats (min, max, mean, standard deviation, median, var, cov) and
    visualize the first 2 columns of the dataset. 
    
    INPUT
    data -- an (N,M) ndarray full of data, which may or may not be clean
    x_name -- string, the name of the feature in column 0, plotted along the X axis
    y_name -- string, the name of the feature in column 1, plotted along the Y axis
    title -- string, the title of the plot
    
    OUTPUT
    min --  an (M,) ndarray of minimum values from each data column
    max --  an (M,) ndarray of maximum values from each data column
    median --  an (M,) ndarray of medians from each data column
    mean --  an (M,) ndarray of means from each data column
    std --  an (M,) ndarray of standard deviations from each data column
    var --  an (M,) ndarray of variances from each data column
    cov --  an (M,M) ndarray of covariances from each pair of data columns'''

    min = np.min( data, axis=0 )
    max = np.max( data, axis=0 )
    median = np.median( data, axis=0 )
    mean = np.mean( data, axis=0 )
    std  = np.std( data, axis=0 ) 
    var = np.var( data, axis=0 )
    cov = np.cov( data, rowvar=True )

    print("raw min:", min)
    print("raw max:", max)
    print("raw median:", median)
    print("raw mean:", mean)
    print("raw std:", std)
    print("raw var:", std)
    print("raw cov:")
    print( cov )

    plt.figure()
    plt.plot( data[:,0], data[:,1], 'ok', label='data' )
    plt.plot( mean[0], mean[1], 'md', label="mean" )
    plt.plot( median[0], median[1], 'cd', label="median" )
    plt.xlabel( x_name )
    plt.ylabel( y_name )
    plt.title( title )
    plt.grid()
    plt.legend()
    plt.pause(0.001)  # Makes the figure window display without blocking, like plt.show().

    return min, max, median, mean, std, var, cov


def main():

    # 1. Read in the Goldie weather buoy data from a file into data matrix named goldie
    goldie, feature_names, feature_units = read_Goldie_from_file()

    # 2. Print just the first few rows of data in columns 0 (date), 4 (air temp), anbd 10 (water temp at 1m depth)
    print("\nGOLDIE DATA")
    print( goldie[0:5,[0,4,10]] )

    # 3. Pull the entire columns of air and 1m water temperatures out into their own Nx2 matrix
    raw = goldie[:,[4,10]]
    print("\nRAW DATA")
    print( raw[0:5,:] )
    print(raw.shape)

    # 4. Compute stats & visualize the RAW DATA.
    axis_labels = [ feature_names[4] + "(" + feature_units[4] + ")",
                    feature_names[10] + "(" + feature_units[10] + ")" ]
    explore( raw, x_name=axis_labels[0], y_name=axis_labels[1], title="Goldie Data: Raw" )
    
    # 4. Use Boolean indexing to select only the entirely NUMERIC rows of the dataset: values that are NOT nan
    numeric = raw[~np.isnan(raw).any(axis=1)]
    print("\nNUMERIC DATA")
    print(numeric)
    print(raw.shape)

    # 5. Compute stats and visualize the numeric data
    explore( numeric, x_name=axis_labels[0], y_name=axis_labels[1], title="Goldie Data: Numeric" )

    # 7. Select only the "CLEAN" data by culling rows with erroneous values
    #       -- Before writing any more code, take a minute to talk with your team about how to identify invalid values in this dataset. Here are some conversation starters:
    #       -- Which axes contain invalid values?
    #       -- What do the invalid values look like?
    #       -- How can boolean indexing help you select only the valid rows?
    #       -- Hint: Numpy's any() and/or all() methods may help.
    abs_zero = -273.15
    clean = numeric[~np.less(numeric, abs_zero).any(axis=1)]
    print("\nCLEAN DATA")
    print(clean)
    print(raw.shape)

    # 8. Compute stats and visualize the clean data
    explore( clean, x_name=axis_labels[0], y_name=axis_labels[1], title="Goldie Data: Numeric" )


if __name__=="__main__":
    main()
    plt.show()