import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pandas as pd

epsilion = 1e-6

def plot_gaussian(data, means, covariances, K, responsibilities):
    plt.scatter(data[:, 0], data[:, 1], c=responsibilities.argmax(axis=1), cmap='viridis', s=40, edgecolor='k',
                alpha=0.2, marker='.')
    x, y = np.mgrid[np.min(data[:, 0]):np.max(data[:, 0]):.01, np.min(data[:, 1]):np.max(data[:, 1]):.01]
    positions = np.dstack((x, y))
    for j in range(K):
        rv = multivariate_normal(means[j], covariances[j])
        plt.contour(x, y, rv.pdf(positions), colors='blue', alpha=0.6, linewidths=1)


def animate(data, iteration, means, covariances, responsibilities, n_components):
    K = n_components
    n_samples, n_features = data.shape

    if n_features != 2:
        print("Drawing animation is only supported for 2D data")
        return

    plt.clf()
    plot_gaussian(data, means, covariances, K, responsibilities)
    plt.title("Iteration: {}".format(iteration))
    plt.pause(0.005)

def EM_Algorithm(data, k, max_iter=50):
    """
    EM Algorithm for Gaussian Mixture Model
    :param data: numpy array of shape (n_samples, n_features)
    :param k: number of clusters
    :param max_iter: maximum number of iterations
    """
    n_samples, n_features = data.shape
    means = np.random.rand(k, n_features)
    covariances = np.zeros((k, n_features, n_features))
    # add a small value to the diagonal of each covariance matrix to ensure that it is positive definite
    for i in range(k):
        covariances[i] = np.eye(n_features) + epsilion
    weights = np.ones(k) / k

    plt.ion()
    for i in range(max_iter):
        print("Iteration: ", i + 1)
        # E-step
        responsibilities = np.zeros(shape=(n_samples, k))
        for j in range(k):
            # get probability density function of each sample for each cluster
            pdf = multivariate_normal.pdf(data, mean=means[j], cov=covariances[j], allow_singular=True)
            responsibilities[:, j] = weights[j] * pdf
        # normalize responsibilities
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

        # M-step
        # Nk is the number of samples in cluster k
        Nk = np.sum(responsibilities, axis=0)
        # calculate new means by weighted average of data points
        # np.newaxis is used to increase the dimension of an array by one more dimension
        means = np.dot(responsibilities.T, data) / Nk[:, np.newaxis]

        for j in range(k):
            delta = data - means[j]
            covariances[j] = np.dot(delta.T, delta * responsibilities[:, j][:, np.newaxis]) / Nk[j]
        weights = Nk / n_samples

        for j in range(k):
            covariances[j] += epsilion * np.eye(n_features)

        if (n_features == 2):
            if i%5 == 0:
                animate(data, i + 1, means, covariances, responsibilities, k)
    plt.ioff()

if __name__ == '__main__':
    # take a data file as input
    # data file contains n data points each having m attributes
    # take dataXD.txt as input
    data = pd.read_csv('data2D_B1.txt', header=None, sep=" ")
    data = data.values

    # Run the EM algorithm with k = k* and plot the data points with the estimated GMM.
    k_star = 5
    EM_Algorithm(data, k_star)