import numpy as np
import misc
### Performs grid search and dumps data
def run_model(l_rbf = 0.1, prior_variance = 1):
    # PARAMETERS
    test_proportion = 0.2
    shuffle = False

    ## c - import data
    X = np.loadtxt('X.txt')
    y = np.loadtxt('y.txt')
    #print("Number of ones {}".format(np.sum(y==1)))
    #misc.plot_data(X,y)

    ## d -  split into training and test data
    num_indices = round((1 - test_proportion) * len(y))
    if shuffle:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        test_indices = indices[:num_indices]
        train_indices = indices[num_indices:]
        X_test = X[test_indices, ...]
        X_train = X[train_indices, ...]
        y_test = y[test_indices, ...]
        y_train = y[train_indices, ...]
    else:
        X_train = X[:num_indices,...]
        X_test = X[num_indices:, ...]
        y_train = y[:num_indices, ...]
        y_test = y[num_indices:, ...]

    ## e - train logistic classifier
    Z = X_train
    # Expand inputs
    X_train_rbf = misc.expand_inputs(l_rbf, X_train, Z)
    X_train_rbf_biased = np.concatenate((X_train_rbf, np.ones((X_train_rbf.shape[0], 1))), 1)
    X_test_rbf = misc.expand_inputs(l_rbf, X_test, Z)
    X_test_rbf_biased = np.concatenate((X_test_rbf, np.ones((X_test_rbf.shape[0], 1))), 1)


    w_MAP = misc.weights(X=X_train_rbf_biased, y=y_train, prior_var=prior_variance)
    w_ML = misc.weights(X=X_train_rbf_biased, y=y_train)

    # Get parameters for Laplace approximation
    hessian = misc.hessian(x=X_train_rbf_biased, w=w_MAP, prior_var=prior_variance)
    # The covariance of the laplace approximation is the inverse of the Hessian
    cov_laplace = np.linalg.inv(hessian)

    y_pred_ML = misc.prediction(X_test_rbf, w_ML)
    y_pred_ML = (y_pred_ML > 0.5)

    y_pred_bayes = misc.bayesian_prediction(cov=cov_laplace, w=w_MAP, x=X_test_rbf)
    y_pred_bayes = (y_pred_bayes > 0.5)


    y_pred_ML = misc.prediction(X_test_rbf, w_ML)
    y_pred_ML = (y_pred_ML > 0.5)
    y_pred_bayes = misc.bayesian_prediction(cov=cov_laplace, w=w_MAP, x=X_test_rbf)
    y_pred_bayes = (y_pred_bayes > 0.5)
    y_train_pred_ML = misc.prediction(X_train_rbf, w_ML)
    y_train_pred_ML = (y_train_pred_ML > 0.5)
    y_train_pred_bayes = misc.bayesian_prediction(cov=cov_laplace, w=w_MAP, x=X_train_rbf)
    y_train_pred_bayes = (y_train_pred_bayes > 0.5)

    C_ML_TEST, _ = misc.confusion(y_ground=y_test, y_pred=y_pred_ML)
    C_ML_TRAIN, _ = misc.confusion(y_ground=y_train, y_pred=y_train_pred_ML)
    C_BAYES_TEST, _ = misc.confusion(y_ground=y_test, y_pred=y_pred_bayes)
    C_BAYES_TRAIN, _ = misc.confusion(y_ground=y_train, y_pred=y_train_pred_bayes)

    data = np.zeros(24,dtype=np.float)
    data[0] = l_rbf
    data[1] = prior_variance
    data[2:6] = C_ML_TEST.ravel()        #MAP TEST
    data[6] = misc.compute_average_ll(w=w_ML, X=X_test_rbf, y=y_test)
    data[7:11] = C_ML_TRAIN.ravel()
    data[11] = misc.compute_average_ll(w=w_ML, X=X_train_rbf, y=y_train)
    data[12:16] = C_BAYES_TEST.ravel()
    data[16] = misc.compute_average_ll(w=w_MAP, X=X_test_rbf, y=y_test)
    data[17] = misc.compute_average_ll(w=w_MAP, X=X_test_rbf, y=y_test, cov=cov_laplace)
    data[18:22] = C_BAYES_TRAIN.ravel()
    #data[22] = misc.compute_average_ll(w=w_MAP, X=X_train_rbf, y=y_train)

    sig_test = misc.logistic(X_test_rbf_biased @ w_MAP)
    sig_test = np.dot(y_test,sig_test) + np.dot(1-y_test,1-sig_test)

    sig_train = misc.logistic(X_train_rbf_biased @ w_MAP)
    sig_train = np.dot(y_train, sig_train) + np.dot(1 - y_train, 1 - sig_train)

    # print("2 - {}".format(sig))
    prior= (1/2*prior_variance)*np.dot(w_MAP,w_MAP)
    # print("3 - {}".format(sig))
    prior+=0.5*len(w_MAP)*np.log(np.pi/2)
    # print("4 - {}".format(sig))
    chol = np.linalg.cholesky(hessian)
    # print("Cholesky det: {}".format(np.log(np.linalg.det(chol))))
    prior-=2*np.sum(np.log(np.diag(chol)))
    # print("5 - {}".format(sig))

    data[22] = sig_test + prior
    data[23] = sig_train + prior

    return data



num_tests = 10
l_vals = np.logspace(-2, 2, num=num_tests+1)
sig_vals = np.logspace(-2, 2, num=num_tests+1)

i = 0
max_runs = len(sig_vals)*len(l_vals)
d = np.zeros((max_runs, 24), np.float)
for l in l_vals:
    for sig in sig_vals:
        print("Run {} of {}: l = {}, σ = {}".format(i+1,max_runs, l, sig))
        d[i, :] = run_model(l, sig)
        i+=1

np.savetxt('rbf_dump_det.csv', d, delimiter=",")