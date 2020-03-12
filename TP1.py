import numpy as np
from numpy.linalg import norm
from scipy.linalg import toeplitz
from numpy.random import multivariate_normal
from scipy.linalg.special_matrices import toeplitz
from numpy.random import randn

def simu_linreg(coefs, n_samples=1000, corr=0.5):
    """Simulation of a linear regression model
    
    Parameters
    ----------
    coefs : `numpy.array`, shape (n_features,)
        Coefficients of the model
    
    n_samples : `int`, default=1000
        Number of samples to simulate
    
    corr : `float`, default=0.5
        Correlation of the features

    Returns
    -------
    A : `numpy.ndarray`, shape (n_samples, n_features)
        Simulated features matrix. It samples of a centered Gaussian 
        vector with covariance given by the Toeplitz matrix
    
    b : `numpy.array`, shape (n_samples,)
        Simulated labels
    """
    n_features = len(coefs)
    # Construction of a covariance matrix
    cov = toeplitz(corr ** np.arange(0, n_features))
    # Simulation of features
    A = multivariate_normal(np.zeros(n_features), cov, size=n_samples)
    # Simulation of the labels
    b = A.dot(coefs) + randn(n_samples)
    return A, b

def TP1(A, b, s=0.01, n_iter=1000, 
        rtol=1e-10, verbose=True):
    """Proximal gradient descent algorithm
    ------------------------------------------------
    Solves the minimization problem :
    min 0.5 || Ax - b ||_2^2 / n_samples + s ||x||_1
    ------------------------------------------------
    prameters :
    
    A : numpy array (n_samples * n_features)
    b : numpy array (n_samples)
    s : a positif float(Lasso's regularization)
    rtol : minimum tolerance
    n_iter : maximum number of iterations
    verbose : when true prints things 
    ------------------------------------------------
    returns :
    
    x : the optimal solution found
    obj : a numpy array [ f(x_k) for k = 1...till convergence]
    precision : a numpy array [prec_k = norm(x_{k+1} - x_k)]
    
    """
    n_samples, n_features = A.shape
    
    def g(x, s):
        """Value of the Lasso penalization at x"""
        return s*norm(x, ord=1)
    
    def f(x):
        """Least-squares loss"""
        return norm(A @ x - b, ord=2)**2/ 2 / n_samples
    
    def prox_g(x, s, step):
        """Proximal operator for the Lasso at x"""    
        return np.sign(x) * np.maximum(np.abs(x) - s * step,0)

    def grad_f(x):
        """Least_squares grad"""
        return A.T @ (A @ x - b) / n_samples
    
    def lip_(A):
        """Lipschitz constant for linear squares loss"""    
        return norm(A, ord=2)**2 / n_samples

    step = 1 / lip_(A)
    
    x = np.zeros(n_features)
    x_new = x.copy()

    # estimation error history
    precision = []
    # objective history
    objectives = []
    # Current estimation error
#     errors.append(err)
    # Current objective
    obj = f(x) + g(x, s)
    objectives.append(obj)
    if verbose:
        print("Lauching PGD solver...")
        print(' | '.join([name.center(8) for name in ["it", "obj", "prec"]]))
    for k in range(n_iter + 1):
        x_new = prox_g(x - step * grad_f(x), s, step)
        
        prec = norm(x_new - x) / max(1, norm(x_new))
        x = x_new
        obj = f(x) + g(x, s)
    
        precision.append(prec)
        objectives.append(obj)
        
        if k % 10 == 0 and verbose:
            print(' | '.join([("%d" % k).rjust(8), 
                              ("%.2e" % obj).rjust(8), 
                              ("%.2e" % prec).rjust(8)]))
        if prec <= rtol:
            print("converged")
            break
            
        if k == n_iter :
            print("algorithm did not converge, increase N_iter ")
            
    return x, np.array(objectives), np.array(precision)

if __name__=="__main__":
    coefs = np.random.choice([0, 4, 40], size=50, p= [.8, .1, .1])
    A, b = simu_linreg(coefs)
    x_opt = TP1(A, b, s=s, n_iter=1000, verbose=True)[0]