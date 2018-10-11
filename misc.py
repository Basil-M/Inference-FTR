import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import fmin_l_bfgs_b as sp_minimise
from scipy.optimize import minimize as sp_min2


##
# X: 2d array with the input features
# y: 1d array with the class labels(0 or 1)
#
def plot_data_internal(X, y, title, sub=None):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), \
                         np.linspace(y_min, y_max, 100))
    if sub is None:
        plt.figure()
    else:
        plt.subplot(sub)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label='Class 0', ms=2)
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label='Class 1', ms=2)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
    return xx, yy


##
# X: 2d array with the input features
# y: 1d array with the class labels(0 or 1)
#
def plot_data(X, y):
    xx, yy = plot_data_internal(X, y)
    plt.show()


##
# x: input to the logistic function
#
def logistic(x): return 1.0 / (1.0 + np.exp(-x))


##
def compute_ll(X, y, w, cov=None):
    x_tilde = np.concatenate((X, np.ones((X.shape[0], 1))), 1)
    mean = x_tilde @ w
    if cov is None:
        prob = logistic(mean)
    else:
        var_a = np.diag(x_tilde @ (cov @ x_tilde.T))
        k_probit = (1 + 0.125 * np.pi * var_a) ** -0.5
        prob = logistic(k_probit * mean)

    # if np.sum(prob == 1) + np.sum(prob ==0) > 0:
    #     print("We have some deterministic bois! \nNumber of prob 1s: {} \nNumber of prob 0s: {}".format(np.sum(prob == 1), np.sum(prob==0)))

    return y * np.log(prob) + (1 - y) * np.log(1.0 - prob)


##
# X: 2d array with the input features
# y: 1d array with the class labels(0 or 1)
# w: current parameter values
#
def compute_average_ll(X, y, w, cov=None):
    return np.mean(compute_ll(X, y, w, cov))  ##


# ll: 1d array with the average likelihood per data point and dimension equal
# to the number of training epochs.
#
def plot_ll(ll):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps ')
    plt.ylabel('Average log - likelihood')
    plt.title('Plot Average Log - likelihood Curve')
    plt.show()


# ll: 1d array with the average likelihood per data point and dimension equal
# to the number of training epochs.
#
def plot_lls(ll_test, ll_train, plot_title):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll_test) + 2)
    plt.ylim(min(ll_train) - 0.1, max(ll_train) + 0.1)
    test_plot, = ax.plot(np.arange(1, len(ll_test) + 1), ll_test, 'g-', label='Test dataset')
    train_plot, = ax.plot(np.arange(1, len(ll_train) + 1), ll_train, 'b-', label='Training dataset')
    plt.xlabel('Steps ')
    plt.ylabel('Average log-likelihood ')
    plt.title(plot_title)
    plt.legend(handles=[test_plot, train_plot])
    plt.show()


# ll: 1d array with the average likelihood per data point and dimension equal
# to the number of training epochs.
#
def plot_ll_rate(ll, rates):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(np.min(ll) - 0.1, np.max(ll) + 0.1)
    for i in range(ll.shape[1]):
        ax.plot(np.arange(1, len(ll) + 1), ll[:, i], label='Rate = {}'.format(rates[i]))
    plt.xlabel('Steps ')
    plt.ylabel('Average log-likelihood ')
    plt.title('Log likelihood convergence by rate')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels)
    plt.show()


##
# x: 2d array with input features at which to compute predictions.
# #( uses parameter vector w which is defined outside the function 's scope )
#
def predict_for_plot(w):
    def predict_given_w(x):
        x_tilde = np.concatenate((x, np.ones((x.shape[0], 1))), 1)
        return logistic(np.dot(x_tilde, w))

    return predict_given_w


##
# X: 2d array with the input features
# y: 1d array with the class labels(0 or 1)
# predict : function that recives as input a feature matrix and returns a 1d
# vector with the probability of class 1.
def plot_predictive_distribution(X, y, predict, sub=None, title='Plot data'):
    xx, yy = plot_data_internal(X, y, sub=sub, title=title)
    ax = plt.gca()

    X_predict = np.concatenate((xx.ravel().reshape((-1, 1)), \
                                yy.ravel().reshape((-1, 1))), 1)
    Z = predict(X_predict)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap='seismic_r', linewidths=1)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=10)
    plt.imshow(Z, interpolation='bilinear', origin='lower',
               cmap='seismic_r', alpha=0.5, extent=(np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)), zorder=0)
    if sub is None:
        plt.show()


##
# l: hyper - parameter for the width of the Gaussian basis functions
# Z: location of the Gaussian basis functions
# X: points at which to evaluate the basis functions
def expand_inputs(l, X, Z):
    X2 = np.sum(X ** 2, 1)
    Z2 = np.sum(Z ** 2, 1)
    ones_Z = np.ones(Z.shape[0])
    ones_X = np.ones(X.shape[0])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l ** 2 * r2)


##
# x: 2d array with input features at which to compute the predictions
# using the feature expansion
# #( uses parameter vector w and the 2d array X with the centers of the basis
# functions for the feature expansion , which are defined outside the function 's
# scope )
#
def predict_for_plot_expanded_features(w, l, z):
    def predict_given_w(x):
        x_expanded = expand_inputs(l, x, z)
        x_tilde = np.concatenate((x_expanded, np.ones((x_expanded.shape[0], 1))), 1)
        return logistic(np.dot(x_tilde, w))

    return predict_given_w


def BatchGenerator(X, y, batch_size):
    cInd = 0
    indices = np.arange(len(y))
    np.random.shuffle(indices)
    X = X[indices, ...]
    y = y[indices]
    while cInd < len(y):
        if cInd + batch_size < len(y):
            yield X[cInd:cInd + batch_size, ...], y[cInd:cInd + batch_size]
        else:
            yield X[cInd:, ...], y[cInd:]
        cInd += batch_size


def confusion(y_ground, y_pred, cout=False):
    C = confusion_matrix(y_ground, y_pred)
    if cout:
        tn, fp, fn, tp = C.ravel()
        print("Confusion matrix:")
        print("\t Predicted label \t")
        print("\t   \t\t 0 \t\t 1")
        print("\t\t 0 \t {:.2f} \t {:.2f}".format(tn / (tn + fp), fp / (tn + fp)))
        print("True label")
        print("\t\t 1 \t {:.2f} \t {:.2f}".format(fn / (fn + tp), tp / (fn + tp)))

    # C = C.astype(np.double)
    # C[0,:] = C[0,:]/(tn + fp)
    # C[1,:] = C[1,:]/(fn + tp)
    return C, np.sum(y_pred)


###FUNCTIONS FOR FTR
def posterior(w, *args):
    prior_var = args[0]
    X = args[1]
    y = args[2]

    # contribution from likelihood
    sig = logistic(X @ w)
    L = np.dot(y, np.log(sig)) + np.dot((1 - y), np.log(1 - sig))

    # if prior variance provided, include term due to prior
    if prior_var is not None:
        # assumed prior is an uncorrelated gaussian with same variance for all parameters
        L += -(0.5 / prior_var) * w.T @ w


    return -1 * L


def posterior_grad(w, *args):
    prior_var = args[0]
    X = args[1]
    y = args[2]

    # contribution from likelihood
    dL = np.dot((y - logistic(X @ w)), X)

    # if prior variance provided, include term due to prior
    if not prior_var is None:
        dL += -(1 / prior_var) * w

    return -1 * dL


def weights(X, y, prior_var=None):
    if prior_var is None:
        w0 = np.random.normal(0, 1, X.shape[1])
        max_iter = 500
        # return ML_weights(max_iter,X=X,y=y)
    else:
        w0 = np.random.normal(0, 1, X.shape[1])
        max_iter = 500

    # out = sp_min2(fun=posterior,
    #               x0=w0,
    #               args=(prior_var, X, y),
    #               jac=posterior_grad,
    #               method='TNC')
    #
    # return out['x']
    out = sp_minimise(func=posterior,
                      x0=w0,
                      fprime=posterior_grad,
                      args=(prior_var, X, y),
                      maxiter=max_iter)

    print(out[2]['nit'])
    return out[0]


def prediction(x, w):
    x_t = np.concatenate((x, np.ones((x.shape[0], 1))), 1)
    return logistic(x_t @ w)


def hessian(prior_var, x, w):
    # Component due to prior
    ## Assumes uncorrelated weights i.e. diagonal covariance matrix
    hess = np.eye(x.shape[1]) / prior_var

    # Calculate logistic function
    sig = logistic(x @ w)

    # Component due to data
    for row, yn in zip(x, sig * (1 - sig)):
        hess += yn * np.outer(row, row)

    return hess


def bayesian_prediction(cov, x, w):
    # Bias inputs
    x_tilde = np.concatenate((x, np.ones((x.shape[0], 1))), 1)

    # Calculate mean and variance of Laplace approximation
    ## Note: Covariance is the inverse of the Hessian
    mean = x_tilde @ w
    var_a = np.diag(x_tilde @ (cov @ x_tilde.T))

    # Approximate logistic with probit for integral over posterior.
    ## This gives another probit. Approximate that probit with logistic.
    k_probit = (1 + 0.125 * np.pi * var_a) ** -0.5
    return logistic(k_probit * mean)


def predict_for_plot_bayesian(cov, w):
    def predict_given_w(x):
        return bayesian_prediction(cov, x, w)

    return predict_given_w


def predict_for_plot_expanded_bayesian(cov, w, l, Z):
    def predict_given_w(x):
        x_expanded = expand_inputs(l, x, Z)
        return bayesian_prediction(cov, x_expanded, w)

    return predict_given_w


def plot_all(X, y, w_ML, w_MAP, cov, prior_var=None, l=None, Z=None):
    if Z is None or l is None:
        plot_predictive_distribution(X, y, predict_for_plot(w_ML), sub=131, title='Maximum Likelihood')
        plot_predictive_distribution(X, y, predict_for_plot(w_MAP, l, Z), sub=132, title='Maximum A-Posteriori')
        plot_predictive_distribution(X, y, predict_for_plot_bayesian(cov, w_MAP), sub=133,
                                     title='MAP, bayesian contours')
    else:
        plot_predictive_distribution(X, y, predict_for_plot_expanded_features(w_ML, l, Z), sub=131,
                                     title='ML, l = {}'.format(l))
        plot_predictive_distribution(X, y, predict_for_plot_expanded_features(w_MAP, l, Z), sub=132,
                                     title='MAP, l = {}, σ = {}'.format(l, prior_var))
        plot_predictive_distribution(X, y, predict_for_plot_expanded_bayesian(cov, w_MAP, l, Z), sub=133,
                                     title='MAP, bayesian contours')

    fig = plt.gcf()
    fig.set_size_inches(15, 7)

    plt.show()


def n_mean(val):
    val = val[val != np.inf]
    val = val[val != -np.inf]
    return np.nanmean(val)


def ML_weights(max_iter, y, X):
    n_iter = 0
    w = np.random.normal(0, 1, X.shape[1])

    while n_iter < max_iter:
        n_iter += 1
        # print("Epoch {}. Current diff: {}".format(n_epochs, LL_change))
        for [y_batch, X_batch] in BatchGenerator(y, X, 50):
            sig = logistic(X_batch @ w)
            error = np.dot((y_batch - sig), X_batch)
            w += 0.001 * error

    return w


def model_evidence(x, y, w, hess, prior_variance):
    # Compute likelihood (f(x0))
    sig = logistic(x @ w)
    likelihood_ev = np.dot(y, np.log(sig)) + np.dot((1 - y), np.log(1 - sig))

    ## terms due to prior - N.B. terms in pi cancel out later, so not included
    prior_ev = -(0.5 / prior_variance) * w.T @ w
    prior_ev -= 0.5*len(w)*np.log(prior_variance)

    # Cholesky decomposition of Hessian to avoid numerical errors
    chol = np.linalg.cholesky(hess)
    log_det_A_inv = -2*np.sum(np.log(np.diag(chol)))

    return likelihood_ev + prior_ev + 0.5*log_det_A_inv

