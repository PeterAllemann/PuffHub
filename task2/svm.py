import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os

TRAIN_DATA = np.array([])
TEST_DATA = np.array([])
TRAIN_LABELS = np.array([])
TEST_LABELS = np.array([])
C = 1.0

C_RANGE = np.logspace(-5, 1, 7)
GAMMA_RANGE = np.logspace(-5, 1, 7)
DEGREE_RANGE = [1, 2, 3]


def initialize_data():
    """
    Initialize the data from MNIST test and train set.
    """

    file_path = os.path.dirname(os.path.abspath(__file__))
    trainData_path = os.path.join(file_path, 'task2/data/mnist-csv/mnist_train.csv')
    testData_path = os.path.join(file_path, 'task2/data/mnist-csv/mnist_test.csv')


    train_set = pd.read_csv(trainData_path, header=None)
    test_set = pd.read_csv(testData_path, header=None)

    # build train and test set without the labels
    global TRAIN_DATA, TEST_DATA
    TRAIN_DATA = train_set.iloc[:, 1:784].values
    TEST_DATA = test_set.iloc[:, 1:784].values

    # get the labels
    global TRAIN_LABELS, TEST_LABELS
    TRAIN_LABELS = train_set.iloc[:, 0].values
    TEST_LABELS = test_set.iloc[:, 0].values


def linear_svm():
    """
    Compute SVM with linear kernel using cross validation and
    parameter tuning to find the best parameter C.
    """

    print("### Linear kernel")

    tuned_parameters = [{'kernel': ['linear'], 'C': C_RANGE}]
    score = 'accuracy'

    print("# Tuning hyper-parameters for %s" % score)
    print()

    svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring=score, iid=False, cv=3, return_train_score=True)
    svc.fit(TRAIN_DATA[:30000], TRAIN_LABELS[:30000])

    print("Best parameters set found on development set:")
    print()
    print(svc.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    means = svc.cv_results_['mean_test_score']
    stds = svc.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, svc.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    predicted = svc.predict(TEST_DATA)
    accuracy = accuracy_score(TEST_LABELS, predicted)
    print('Accuracy on test set: {:.4f}'.format(accuracy))


def polynomial_kernel():
    """
    Compute SVM with polynomial kernel using cross validation and
    parameter tuning to find the best parameters C, gamma and degree.
    """

    print("### Polynomial kernel")

    tuned_parameters = [{'kernel': ['poly'], 'gamma': GAMMA_RANGE, 'degree': DEGREE_RANGE, 'C': C_RANGE}]
    score = 'accuracy'

    print("# Tuning hyper-parameters for %s" % score)
    print()

    poly_svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring=score, iid=False, cv=3, return_train_score=True)
    poly_svc.fit(TRAIN_DATA[:30000], TRAIN_LABELS[:30000])

    print("Best parameters set found on development set:")
    print()
    print(poly_svc.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    means = poly_svc.cv_results_['mean_test_score']
    stds = poly_svc.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, poly_svc.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    predicted = poly_svc.predict(TEST_DATA)
    accuracy = accuracy_score(TEST_LABELS, predicted)
    print('Accuracy on test set: {:.4f}'.format(accuracy))


def rbf_kernel():
    """
    Compute SVM with RBF kernel using cross validation and
    parameter tuning to find the best parameters C and gamma.
    """

    print("### RBF kernel")

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': GAMMA_RANGE, 'C': C_RANGE}]
    score = 'accuracy'

    print("# Tuning hyper-parameters for %s" % score)
    print()

    rbf_svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring=score, iid=False, cv=3, return_train_score=True)
    rbf_svc.fit(TRAIN_DATA[:30000], TRAIN_LABELS[:30000])

    print("Best parameters set found on development set:")
    print()
    print(rbf_svc.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    means = rbf_svc.cv_results_['mean_test_score']
    stds = rbf_svc.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, rbf_svc.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    predicted = rbf_svc.predict(TEST_DATA)
    accuracy = accuracy_score(TEST_LABELS, predicted)
    print('Accuracy on test set: {:.4f}'.format(accuracy))


def combined_computation():
    """
    Compute SVM using cross validation and parameter tuning to
    find the best kernel (RBF, linear or polynomial) and the
    corresponding best parameters.
    """

    print("### Combined")

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': GAMMA_RANGE, 'C': C_RANGE},
                        {'kernel': ['linear'], 'C': C_RANGE},
                        {'kernel': ['poly'], 'gamma': GAMMA_RANGE, 'degree': DEGREE_RANGE, 'C': C_RANGE}]
    score = 'accuracy'

    print("# Tuning hyper-parameters for %s" % score)
    print()

    svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring=score, iid=False, cv=3, return_train_score=True)
    svc.fit(TRAIN_DATA[:30000], TRAIN_LABELS[:30000])

    print("Best parameters set found on development set:")
    print()
    print(svc.best_params_)
    print()
    print("Grid scores on development set:")
    print()

    means = svc.cv_results_['mean_test_score']
    stds = svc.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, svc.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    predicted = svc.predict(TEST_DATA)
    accuracy = accuracy_score(TEST_LABELS, predicted)
    print('Accuracy on test set: {:.4f}'.format(accuracy))


initialize_data()
linear_svm()
polynomial_kernel()
rbf_kernel()
combined_computation()
