import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

TRAIN_DATA = np.array([])
VAL_DATA = np.array([])
TEST_DATA = np.array([])

TRAIN_LABELS = np.array([])
VAL_LABELS = np.array([])
TEST_LABELS = np.array([])
C = 1.0

C_RANGE = np.logspace(-5, 1, 7)
GAMMA_RANGE = np.logspace(-5, 1, 7)
DEGREE_RANGE = [1, 2, 3]


def initialize_data():
    """
    Initialize the data from MNIST test and train set.
    """

    train_set = pd.read_csv('DATA/mnist-csv/mnist_train.csv', header=None)
    test_set = pd.read_csv('DATA/mnist-csv/mnist_test.csv', header=None)

    # build train and test set without the labels
    global TRAIN_DATA, TEST_DATA
    data = train_set.iloc[:, 1:784].values
    TEST_DATA = test_set.iloc[:, 1:784].values

    # get the labels
    global TRAIN_LABELS, TEST_LABELS
    labels = train_set.iloc[:, 0].values
    TEST_LABELS = test_set.iloc[:, 0].values

    n_train = 50000

    TRAIN_DATA = data[:n_train]         # 50000
    TRAIN_LABELS = labels[:n_train]     # 50000

    global VAL_DATA, VAL_LABELS
    VAL_DATA = data[n_train:]           # 10000
    VAL_LABELS = labels[n_train:]       # 10000


def linear_svm():
    """
    Compute SVM with linear kernel using cross validation and
    parameter tuning to find the best parameter C.
    """

    print("### Linear kernel")

    tuned_parameters = [{'kernel': ['linear'], 'C': C_RANGE}]
    best_params, svc = tune_parameters('linear kernel', tuned_parameters)

    print("# Best parameters set found on validation set:")
    print(best_params)

    print_grid_scores(svc)

    print("# Fit SVM with train set:")
    svc_optimized_linear = svm.SVC(kernel='linear', C=best_params['C'])
    svc_optimized_linear.fit(TRAIN_DATA, TRAIN_LABELS)

    predict(svc_optimized_linear)


def polynomial_kernel():
    """
    Compute SVM with polynomial kernel using cross validation and
    parameter tuning to find the best parameters C, gamma and degree.
    """

    print("### Polynomial kernel")

    tuned_parameters = [{'kernel': ['poly'], 'gamma': GAMMA_RANGE, 'degree': DEGREE_RANGE, 'C': C_RANGE}]
    best_params, svc = tune_parameters('polynomial kernel', tuned_parameters)

    print("# Best parameters set found on validation set:")
    print(best_params)

    print_grid_scores(svc)

    print("# Fit SVM with train set:")
    svc_optimized_poly = svm.SVC(kernel='poly', C=best_params['C'], degree=best_params['degree'], gamma=best_params['gamma'])
    svc_optimized_poly.fit(TRAIN_DATA, TRAIN_LABELS)

    predict(svc_optimized_poly)


def rbf_kernel():
    """
    Compute SVM with RBF kernel using cross validation and
    parameter tuning to find the best parameters C and gamma.
    """

    print("### RBF kernel")
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': GAMMA_RANGE, 'C': C_RANGE}]

    best_params, svc = tune_parameters('RBF kernel', tuned_parameters)

    print("# Best parameters set found on development set:")
    print(best_params)

    print_grid_scores(svc)

    print("# Fit SVM with train set:")
    svc_optimized_rbf = svm.SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
    svc_optimized_rbf.fit(TRAIN_DATA, TRAIN_LABELS)

    predict(svc_optimized_rbf)


def tune_parameters(model, tuned_parameters):
    print("# Tuning hyper-parameters for %s" % model)

    svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring='accuracy', iid=False, cv=3, return_train_score=True)
    svc.fit(VAL_DATA, VAL_LABELS)

    return svc.best_params_, svc


def print_grid_scores(model):
    print("# Grid scores on validation set:")
    means = model.cv_results_['mean_test_score']
    stds = model.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, model.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()


def predict(model):
    print("# Predict on test set:")
    predicted = model.predict(TEST_DATA)
    accuracy = accuracy_score(TEST_LABELS, predicted)
    print('Accuracy on test set: {:.4f}'.format(accuracy))


initialize_data()
linear_svm()
polynomial_kernel()
rbf_kernel()