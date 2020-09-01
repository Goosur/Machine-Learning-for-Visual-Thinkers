import matplotlib.pyplot as plt
import numpy as np  

def normal():
    # Draw samples from the distribution:
    mu_x , sigma_x = 0, 10
    mu_y , sigma_y = 5, 3
    shape = (1000,)
    x = np.random.normal(mu_x, sigma_x, shape)
    y = np.random.normal(mu_y, sigma_y, shape)

    # Display the histogram of the samples, along with the
    # probability density function:
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.suptitle("Normal Distribution")

    ax[0].set_title("Example Data")
    ax[0].plot(x, y, 'ob', alpha=0.5, markeredgecolor='w', markersize=3)
    ax[0].set_xlabel("Generated x Value")
    ax[0].set_ylabel("Generated y Value")
    ax[0].grid()
    ax[0].axis('equal')

    n_bins = 15
    n_samps = len(x)
    ax[1].set_title("x Density")
    count_x, bins_x, ignored_x = ax[1].hist(x, n_bins, density=True)
    ax[1].plot(bins_x, 1/(sigma_x * np.sqrt(2 * np.pi)) * np.exp( - (bins_x - mu_x)**2 / (2 * sigma_x**2)), linewidth=2, color='r')
    ax[1].set_xlabel("Generated Value")
    ax[1].set_ylabel("Normalized Count")

    n_bins = 15
    n_samps = len(y)
    ax[2].set_title("y Density")
    count_y, bins_y, ignored_y = ax[2].hist(y, n_bins, density=True)
    ax[2].plot(bins_y, 1/(sigma_y * np.sqrt(2 * np.pi)) * np.exp( - (bins_y - mu_y)**2 / (2 * sigma_y**2)), linewidth=2, color='r')
    ax[2].set_xlabel("Generated Value")
    ax[2].set_ylabel("Normalized Count")

    fig, ax = plt.subplots()
    plt.title("Just the 1-dimensional x data:")
    ax.plot(x, np.zeros(x.shape), 'ob', alpha=0.5, markeredgecolor='w')

    plt.show()

def uniform():
    # Draw samples from the distribution:
    low_x = -1
    high_x = 0
    low_y = 100
    high_y = 200
    shape = (1000,)
    x = np.random.uniform(low_x, high_x, shape)
    y = np.random.uniform(low_y, high_y, shape)

    # All values are within the given interval:
    print("all x >=1 ?", np.all(x >= 1))
    print("all x < 0 ?", np.all(x < 0))
    range_x = high_x - low_x
    range_y = high_y - low_y
    
    # Display the histogram of the samples, along with the
    # probability density function:
    fig, ax = plt.subplots(nrows=1, ncols=3)
    plt.suptitle("Uniform Distribution")

    ax[0].set_title("Example Data")
    ax[0].plot(x, y, 'ob', alpha=0.5, markeredgecolor='w', markersize=3)
    ax[0].set_xlabel("Generated x values")
    ax[0].set_ylabel("Generated y values")
    ax[0].grid()

    n_bins = 15
    n_samps = len(x)
    ax[1].set_title("x Density")
    count_x, bins_x, ignored_x = ax[1].hist(x, n_bins, density=True)
    ax[1].plot(bins_x, np.ones_like(bins_x) / range_x, linewidth=2, color='r')
    ax[1].set_xlabel("Generated Value")
    ax[1].set_ylabel("Normalized Count")
    
    n_bins = 15
    n_samps = len(y)
    ax[2].set_title("y Density")
    count_y, bins_y, ignored_y = ax[2].hist(y, n_bins, density=True)
    ax[2].plot(bins_y, np.ones_like(bins_y) / range_y, linewidth=2, color='r')
    ax[2].set_xlabel("Generated Value")
    ax[2].set_ylabel("Normalized Count")

    fig, ax = plt.subplots()
    plt.title("Just the 1-dimensional x data:")
    ax.plot(x, np.zeros(x.shape), 'ob', alpha=0.5, markeredgecolor='w')

    plt.show()

def main():
    normal()
    uniform()

main()