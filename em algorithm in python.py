import numpy as np
from scipy.stats import multivariate_normal
# Define EM algorithm function
def em_algorithm(X, k, max_iter=100):
    """
    Applies the Expectation-Maximization algorithm to estimate the parameters
    of a multivariate Gaussian mixture model.

    Arguments:
    - X: a numpy array of shape (n, d) representing the data points, where n is
         the number of data points and d is the number of dimensions.
    - k: an integer representing the number of components in the mixture model.
    - max_iter: an integer representing the maximum number of iterations of the
                algorithm to perform (default=100).

    Returns:
    - mu: a numpy array of shape (k, d) representing the means of the k components.
    - cov: a numpy array of shape (k, d, d) representing the covariance matrices
           of the k components.
    - pi: a numpy array of shape (k,) representing the mixture coefficients of the
          k components.
    """
    # Function body goes here

    n, d = X.shape
    # Step 1: Initialize the parameters
    mu = np.random.rand(k, d)
    cov = np.array([np.eye(d)] * k)
    pi = np.array([1 / k] * k)
    for iteration in range(max_iter):
        # Step 2: Calculate the responsibilities
        resp = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                resp[i, j] = pi[j] * multivariate_normal.pdf(X[i], mu[j], cov[j])
        resp /= resp.sum(axis=1, keepdims=True)
        # Step 3: Update the parameters
        Nk = resp.sum(axis=0)
        mu = resp.T @ X / Nk[:, None]
        cov = np.zeros((k, d, d))
        for j in range(k):
            diff = X - mu[j, :]
            cov[j] = (resp[:, j, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / Nk[j]
        pi = Nk / n
    return mu, cov, pi
