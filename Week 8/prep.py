# prep.py
#   Load lwfcrop.
# Devon Gardner
# 10/15/2020

import os
import numpy as np
import matplotlib.pyplot as plt

def read_lfwcrop():
    faces_filename = 'lfwcrop.npy'
    current_directory = os.path.dirname(__file__)
    faces_filepath = os.path.join(current_directory, '..', 'data', faces_filename)
    lfw_faces = np.load(faces_filepath)

    names_filename = 'lfwcrop_ids.txt'
    names_filepath = os.path.join(current_directory, '..', 'data', names_filename)
    lfw_names = np.loadtxt(names_filepath, dtype=str, delimiter='\n')

    return lfw_faces, lfw_names


def plot_face(image, title, ax=None):
    pass

def main():
    # Read in the dataset, including all images and the names of the associated people
    X, lfw_names = read_lfwcrop()
    n = X.shape[0]
    m = X.shape[1]*X.shape[2]
    print('faces:', X.shape)
    print('names:', len(lfw_names))
    print('features:', m)

    # Visualize the first face
    first_face = X[0,:,:]
    first_name = lfw_names[0]
    plt.figure()
    plt.imshow(first_face, cmap='bone')
    plt.title(first_name)

    # Visualize a random face
    rand_idx = np.random.randint(n)
    rand_face = X[rand_idx,:,:]
    rand_name = lfw_names[rand_idx]
    plt.figure()
    plt.imshow(rand_face, cmap='bone')
    plt.title(rand_name)

    # Visualize the mean face
    mean_face = np.mean(X, axis=0)
    mean_name = 'Mean Face'
    plt.figure()
    plt.imshow(mean_face, cmap='bone')
    plt.title(mean_name)

    # PCA-Cov?
    n_tiny = 100
    X = X[0:n_tiny,:,:]
    X = X.reshape((n_tiny,m))
    print(X.shape)
    X_diff = X - mean_face.reshape((1,m))
    C = np.cov(X_diff, rowvar=False)
    e, P = np.linalg.eig(C)
    print(P)


if __name__ == '__main__':
    main()
    plt.show()