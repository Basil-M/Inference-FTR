import numpy as np
import misc

# PARAMETERS
test_proportion = 0.25
l_rbf = np.round(10**0,3)
prior_variance = np.round(10**-0.8,3)
lin = False
rbf = True
shuffle = False
w_main = np.array([-0.12069063, 0.72605378, 0.19229557])

## c - import data
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')
# print("Number of ones {}".format(np.sum(y==1)))
# misc.plot_data(X,y)

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
    X_train = X[:num_indices, ...]
    X_test = X[num_indices:, ...]
    y_train = y[:num_indices, ...]
    y_test = y[num_indices:, ...]

X_train_biased = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), 1)

## e - train logistic classifier
if lin:
    w_MAP = misc.weights(x=X_train_biased, y=y_train, prior_var=prior_variance)
    w_ML = misc.weights(x=X_train_biased, y=y_train, prior_var=-1)

    print(w_MAP)
    # Get parameters for Laplace approximation
    hessian = misc.hessian(x=X_train_biased, w=w_MAP, prior_var=prior_variance)
    print(np.mean(hessian))
    print(hessian)

    # The covariance of the laplace approximation is the inverse of the Hessian
    cov_laplace = np.linalg.inv(hessian)
    print(cov_laplace)
    misc.plot_predictive_distribution(X_train, y_train, misc.predict_for_plot(w_MAP))
    misc.plot_predictive_distribution(X_train, y_train, misc.predict_for_plot_bayesian(cov_laplace, w_MAP))

    # predict for testing data
    y_pred_ML = misc.prediction(X_test, w_ML)
    y_pred_ML = (y_pred_ML > 0.5)

    y_pred_bayes = misc.bayesian_prediction(cov_laplace, X_test, w_MAP)
    y_pred_bayes = (y_pred_bayes > 0.5)

    # confusion matrices
    print('MAP Confusion matrix (Testing):')
    C, _ = misc.confusion(y_ground=y_test, y_pred=y_pred_ML, cout=True)

    print('BAYES Confusion matrix (Testing):')
    C, _ = misc.confusion(y_ground=y_test, y_pred=y_pred_bayes, cout=True)

    print('Similarity between two:')
    C, _ = misc.confusion(y_ground=y_pred_ML, y_pred=y_pred_bayes, cout=True)

if rbf:
    Z = X_train
    # Expand inputs
    X_train_rbf = misc.expand_inputs(l_rbf, X_train, Z)
    X_train_rbf_biased = np.concatenate((X_train_rbf, np.ones((X_train_rbf.shape[0], 1))), 1)
    X_test_rbf = misc.expand_inputs(l_rbf, X_test, Z)

    w_MAP = misc.weights(X=X_train_rbf_biased, y=y_train, prior_var=prior_variance)
    w_ML = misc.weights(X=X_train_rbf_biased, y=y_train)

    # Get parameters for Laplace approximation
    hessian = misc.hessian(x=X_train_rbf_biased, w=w_MAP, prior_var=prior_variance)
    # The covariance of the laplace approximation is the inverse of the Hessian
    cov_laplace = np.linalg.inv(hessian)


    # PLOTS
    # misc.plt_predictive_distribution(X_train, y_train, misc.predict_for_plot_expanded_features(w_MAP, l_rbf, Z))
    # misc.plot_predictive_distribution(X_train, y_train,
    #                                   misc.predict_for_plot_expanded_bayesian(cov_laplace, w_MAP, l_rbf, Z))
    misc.plot_all(X_train, y_train, w_MAP=w_MAP, w_ML=w_ML, prior_var=prior_variance, l=l_rbf, Z=Z, cov=cov_laplace)

    y_pred_ML = (misc.prediction(X_test_rbf, w_ML) > 0.5)
    y_pred_MAP = (misc.prediction(X_test_rbf, w_MAP) > 0.5)
    y_pred_bayes = (misc.bayesian_prediction(cov=cov_laplace, w=w_MAP, x=X_test_rbf) > 0.5)
    # confusion matrices
    print('ML Confusion matrix (Testing):')
    C, _ = misc.confusion(y_ground=y_test, y_pred=y_pred_ML, cout=True)
    print('MAP Confusion matrix (Testing):')
    C, _ = misc.confusion(y_ground=y_test, y_pred=y_pred_bayes, cout=True)
    print('BAYES Confusion matrix (Testing):')
    C, _ = misc.confusion(y_ground=y_test, y_pred=y_pred_bayes, cout=True)
    print('Similarity between MAP and ML:')
    C, _ = misc.confusion(y_ground=y_pred_bayes, y_pred=y_pred_MAP, cout=True)
    print("Predictive accuracy: {}".format(np.sum(y_pred_MAP == y_test) / 250))


    y_pred_ML = (misc.prediction(X_train_rbf, w_ML) > 0.5)
    y_pred_MAP = (misc.prediction(X_train_rbf, w_MAP) > 0.5)
    y_pred_bayes = (misc.bayesian_prediction(cov=cov_laplace, w=w_MAP, x=X_train_rbf) > 0.5)
    # confusion matrices
    print('ML Confusion matrix (Training):')
    C, _ = misc.confusion(y_ground=y_train, y_pred=y_pred_ML, cout=True)
    print('MAP Confusion matrix (Training):')
    C, _ = misc.confusion(y_ground=y_train, y_pred=y_pred_bayes, cout=True)
    print('BAYES Confusion matrix (Training):')
    C, _ = misc.confusion(y_ground=y_train, y_pred=y_pred_bayes, cout=True)
    print('Similarity between MAP and Bayes:')
    C, _ = misc.confusion(y_ground=y_pred_bayes, y_pred=y_pred_MAP, cout=True)

    ##
    LL = np.zeros((6,1))
    LL[0] = misc.compute_average_ll(w=w_ML, X=X_train_rbf, y=y_train)
    LL[1] = misc.compute_average_ll(w=w_MAP, X=X_train_rbf, y=y_train)
    LL[2] = misc.compute_average_ll(w=w_MAP, X=X_train_rbf, y=y_train, cov=cov_laplace)
    LL[3] = misc.compute_average_ll(w=w_ML, X=X_test_rbf, y=y_test)
    LL[4] = misc.compute_average_ll(w=w_MAP, X=X_test_rbf, y=y_test)
    LL[5] = misc.compute_average_ll(w=w_MAP, X=X_test_rbf, y=y_test, cov=cov_laplace)

    print("Log likelihoods:")
    print(".\t  ML \t MAP \t BAYES")
    print("Train\t {:.3f} \t {:.3f} \t {:.3f}".format(LL[0][0], LL[1][0], LL[2][0]))
    print("Test \t {:.3f} \t {:.3f} \t {:.3f}".format(LL[3][0], LL[4][0], LL[5][0]))

