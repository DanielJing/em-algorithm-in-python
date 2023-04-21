# Import necessary libraries
import numpy as np
from scipy.stats import multivariate_normal
 # Define EM algorithm function
def em_algorithm(X, k, max_iter=100):
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
