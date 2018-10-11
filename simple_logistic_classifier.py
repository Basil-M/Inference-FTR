import numpy as np
import misc

# PARAMETERS
test_proportion = 0.03
batch_size = 750
training_rate = 1E-4
threshold = 0.0000001 #0.1% fractional change in LL
max_epochs = 7500
l_rbf = 1
lin = True
rbf = False

## c - import data
X = np.loadtxt('X.txt')
y = np.loadtxt('y.txt')
print("Number of ones {}".format(np.sum(y==1)))
#misc.plot_data(X,y)

## d -  split into training and test data
num_indices = round(test_proportion * len(y))

indices = np.arange(len(y))
np.random.shuffle(indices)
test_indices = indices[:num_indices]
train_indices = indices[num_indices:]
X_test = X[test_indices, ...]
X_train = X[train_indices, ...]
X_train_biased = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), 1)

y_test = y[test_indices, ...]
y_train = y[train_indices, ...]
    # num_1 = np.sum(y_train == 1)
    # f = num_1/len(y_train)
    # print("{} training data, {} testing data, fraction of 1s {:.4f}".format(len(X_train), len(X_test), f))




## e - train logistic classifier
if lin:
    # initialise parameters
    w_new = np.random.normal(0,1, X_train_biased.shape[1])
    LL_change = 2 * threshold

    n_epochs = 0

    # track LL
    test_LL_evol = np.zeros(int(max_epochs * len(y_train) / batch_size))
    train_LL_evol = np.zeros_like(test_LL_evol)
    i = 0
    while LL_change > threshold and n_epochs < max_epochs:
        n_epochs += 1
        # print("Epoch {}. Current diff: {}".format(n_epochs, LL_change))
        for [y_batch, X_batch] in misc.BatchGenerator(y_train, X_train_biased, batch_size):
            sig = misc.logistic(np.matmul(X_batch, w_new))
            error = np.dot((y_batch - sig), X_batch)
            w_new += training_rate * error

            # track log likelihood evolution
            test_LL_evol[i] = misc.compute_average_ll(X_test, y_test, w_new)
            train_LL_evol[i] = misc.compute_average_ll(X_train, y_train, w_new)
            if i != 0:
                LL_change = abs((train_LL_evol[i] - train_LL_evol[i - 1])/train_LL_evol[i-1])
            i += 1
    print(w_new)
    misc.plot_lls(test_LL_evol[0:i],train_LL_evol[0:i], plot_title='Linear logistic regression - log likelihoods')
    misc.plot_predictive_distribution(X_train, y_train, misc.predict_for_plot(w_new))

    ###f report final training & test log-likelihoods per datapoint
    print('Training LL')
    C = misc.confusion(y_train, X=X_train, w=w_new)
    print("Final log likelihood: {}".format(train_LL_evol[i-1]))
    print('\nTest LL')
    C = misc.confusion(y_test, X=X_test, w=w_new)
    print("Final log likelihood: {}".format(test_LL_evol[i-1]))

###g expand inputs through radial basis functions
if rbf:
    Z = X_train
    X_train_rbf = misc.expand_inputs(l_rbf, X_train, Z)
    X_train_rbf_biased = np.concatenate((X_train_rbf, np.ones((X_train_rbf.shape[0], 1))), 1)

    X_test_rbf = misc.expand_inputs(l_rbf, X_test, Z)

    ##Retrain
    # initialise parameters
    w_rbf = np.random.normal(0,1, X_train_rbf_biased.shape[1])
    LL_change = 2 * threshold

    n_epochs = 0
    # track LL
    test_LL_evol_rbf = np.zeros(int(max_epochs * len(y_train) / batch_size))
    train_LL_evol_rbf = np.zeros_like(test_LL_evol_rbf)
    i = 0
    while LL_change > threshold and n_epochs < max_epochs:
        n_epochs += 1
        #print("Epoch {}. Current diff: {}".format(n_epochs, LL_change))
        for [y_batch, X_batch] in misc.BatchGenerator(y_train, X_train_rbf_biased, batch_size):
            sig = misc.logistic(np.matmul(X_batch, w_rbf))
            error = np.dot((y_batch - sig), X_batch)
            w_rbf += training_rate * error

            # track log likelihood evolution
            test_LL_evol_rbf[i] = misc.compute_average_ll(X_test_rbf, y_test, w_rbf)
            train_LL_evol_rbf[i] = misc.compute_average_ll(X_train_rbf, y_train, w_rbf)
            if i != 0:
                LL_change = abs((train_LL_evol_rbf[i] - train_LL_evol_rbf[i - 1])/train_LL_evol_rbf[i - 1])
            i += 1

    #misc.plot_ll(test_LL_evol_rbf[0:i])
    misc.plot_lls(test_LL_evol_rbf[0:i], train_LL_evol_rbf[0:i], 'Log-likelihoods rbf')
    misc.plot_predictive_distribution(X_test, y_test, misc.predict_for_plot_expanded_features(w_rbf, l_rbf, Z))
    print('Test confusion matrix:')
    # print('Fraction of class 1 in training data: {:.6f}'.format(np.sum(y_train==1)/len(y_train)))
    C, num_1 = misc.confusion(y_test, X_test_rbf, w_rbf)
    print("Final log likelihood: {}".format(test_LL_evol_rbf[i-1]))

    print('Train confusion matrix:')
    C, num_1 = misc.confusion(y_train, X_train_rbf, w_rbf)
    print("Final log likelihood: {}".format(train_LL_evol_rbf[i-1]))
    #print("Fraction of 1s in training data: \t {}".format(f))
    #print("Fraction of 1s in prediction:\t {}".format(num_1/len(y_train)))
    #print("{}\t{}\t{}\t{}\t{}\t{}".format(f, C[0,0], C[0,1], C[1,0], C[1,1], num_1/len(y_test)))