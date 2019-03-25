import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

train_data = np.array([])
test_data = np.array([])
train_labels = np.array([])
test_labels = np.array([])
C = 1.0

C_range = np.logspace(-5, 1, 7)
gamma_range = np.logspace(-5, 1, 7)
degree_range = [1, 2, 3]


def initialize_data():

    train_set = pd.read_csv('DATA/mnist-csv/mnist_train.csv', header=None)
    test_set = pd.read_csv('DATA/mnist-csv/mnist_test.csv', header=None)

    # build train and test set without the labels
    global train_data, test_data
    train_data = train_set.iloc[:, 1:784].values
    test_data = test_set.iloc[:, 1:784].values

    # get the labels
    global train_labels, test_labels
    train_labels = train_set.iloc[:, 0].values
    test_labels = test_set.iloc[:, 0].values


def linear_svm():

    tuned_parameters = [{'kernel': ['linear'], 'C': C_range}]
    score = 'accuracy'

    print("# Tuning hyper-parameters for %s" % score)
    print()

    svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring=score, iid=False, cv=3, return_train_score=True)
    svc.fit(train_data[:30000], train_labels[:30000])

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

    predicted = svc.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted)
    print('Accuracy on test set: {:.4f}'.format(accuracy))


def polynomial_kernel():

    tuned_parameters = [{'kernel': ['poly'], 'gamma': gamma_range, 'degree': degree_range, 'C': C_range}]
    score = 'accuracy'

    poly_svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring=score, iid=False, cv=3, return_train_score=True)
    poly_svc.fit(train_data[:30000], train_labels[:30000])


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

    predicted = poly_svc.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted)
    print('Accuracy on test set: {:.4f}'.format(accuracy))


def rbf_kernel():
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range}]
    score = 'accuracy'

    rbf_svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring=score, iid=False, cv=3, return_train_score=True)
    rbf_svc.fit(train_data[:30000], train_labels[:30000])

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

    predicted = rbf_svc.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted)
    print('Accuracy on test set: {:.4f}'.format(accuracy))


def combined_computation():

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range},
                        {'kernel': ['linear'], 'C': C_range},
                        {'kernel': ['poly'], 'gamma': gamma_range, 'degree': degree_range, 'C': C_range}]
    score = 'accuracy'

    svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring=score, iid=False, cv=3, return_train_score=True)
    svc.fit(train_data[:10000], train_labels[:10000])

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

    predicted = svc.predict(test_data)
    accuracy = accuracy_score(test_labels, predicted)
    print('Accuracy on test set: {:.4f}'.format(accuracy))


initialize_data()
linear_svm()
polynomial_kernel()
rbf_kernel()
combined_computation()
