import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pandas as pd

epsilion = 1e-6

def EM_Algorithm(data, k, max_iter=30):
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

    # Calculate log likelihood
    pdfs = np.zeros((n_samples, k))
    for j in range(k):
        pdfs[:, j] = multivariate_normal.pdf(data, mean=means[j], cov=covariances[j], allow_singular=True)
    log_likelihood = np.sum(np.log(np.sum(pdfs * weights, axis=1)))

    return means, covariances, weights, log_likelihood

if __name__ == '__main__':
    # take a data file as input
    # data file contains n data points each having m attributes
    # take data2D.txt as input
    data = pd.read_csv('data3D.txt', header=None, sep=" ")
    data = data.values

    log_likelihoods = []

    # As the number of components (or, the number of gaussian distributions, k) is usually
    # unknown, you will assume a range for k. For example, from 1 to 10.
    # For each k, you will run the EM algorithm to estimate the GMM, keep a note of the log-likelihood
    k_min = 1
    k_max = 10
    k_range = range(k_min, k_max+1)

    for i in k_range:
        means, covariances, weights, log_likelihood = EM_Algorithm(data, i)
        view = False
        if view == True:
            print('k = ', i)
            print('means = ', means)
            print('covariances = ', covariances)
            print('weights = ', weights)
        log_likelihoods.append(log_likelihood)

    # Show a plot of how converged log-likelihood varies with the number of components k.
    plt.plot(k_range, log_likelihoods)
    plt.xlabel('k')
    plt.ylabel('log-likelihood')
    plt.show()

    # Choose an appropriate value for k from the plot. Let's call it k*.
    k_star = k_range[np.argmax(log_likelihoods)]
    print('k* = ', k_star)