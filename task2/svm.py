import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV

train_data = np.array([])
test_data = np.array([])
train_labels = np.array([])
test_labels = np.array([])
C = 1.0

C_range = np.logspace(-3, 1, 5)
gamma_range = np.logspace(-3, 1, 5)
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

    svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring='accuracy', iid=False, cv=3, return_train_score=True)
    svc.fit(train_data[:10000], train_labels[:10000])

    print("Best parameters set found on development set for linear kernel:")
    print(svc.best_params_)

    # df = pd.DataFrame.from_dict(svc.cv_results_)
    # print(df)

    mean_train_score = svc.cv_results_['mean_train_score']
    mean_test_score = svc.cv_results_['mean_test_score']
    print('mean train accuracy: {}'.format(mean_train_score))
    print('mean test accuracy: {}'.format(mean_test_score))

    score = svc.score(test_data[:500], test_labels[:500])
    print('linear svm score: {:.4f}'.format(score))
    print()


def rbf_kernel():
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range}]

    rbf_svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring='accuracy', iid=False, cv=3, return_train_score=True)
    rbf_svc.fit(train_data[:10000], train_labels[:10000])

    print("Best parameters set found on development set for linear kernel:")
    print(rbf_svc.best_params_)

    # df = pd.DataFrame.from_dict(rbf_svc.cv_results_)
    # print(df)

    mean_train_score = rbf_svc.cv_results_['mean_train_score']
    mean_test_score = rbf_svc.cv_results_['mean_test_score']
    print('mean train accuracy: {}'.format(mean_train_score))
    print('mean test accuracy: {}'.format(mean_test_score))

    rbf_svc_score = rbf_svc.score(test_data[:500], test_labels[:500])
    print('rbf kernel score: {:.4f}'.format(rbf_svc_score))
    print()


def polynomial_kernel():
    tuned_parameters = [{'kernel': ['poly'], 'gamma': gamma_range, 'degree': degree_range, 'C': C_range}]

    poly_svc = GridSearchCV(svm.SVC(), tuned_parameters, scoring='accuracy', iid=False, cv=3, return_train_score=True)
    poly_svc.fit(train_data[:10000], train_labels[:10000])

    print("Best parameters set found on development set for linear kernel:")
    print(poly_svc.best_params_)

    # df = pd.DataFrame.from_dict(poly_svc.cv_results_)
    # print(df)

    mean_train_score = poly_svc.cv_results_['mean_train_score']
    mean_test_score = poly_svc.cv_results_['mean_test_score']
    print('mean train accuracy: {}'.format(mean_train_score))
    print('mean test accuracy: {}'.format(mean_test_score))

    poly_svc_score = poly_svc.score(test_data[:500], test_labels[:500])
    print('polynomial kernel score: {:.4f}'.format(poly_svc_score))
    print()


def combined_computation():
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': C_range},
                        {'kernel': ['linear'], 'C': C_range},
                        {'kernel': ['poly'], 'gamma': gamma_range, 'degree': degree_range, 'C': C_range}]

    clf = GridSearchCV(svm.SVC(), tuned_parameters, scoring='accuracy', iid=False, cv=3, return_train_score=True)
    clf.fit(train_data[:10000], train_labels[:10000])

    print("Best parameters set found on development set:")
    print(clf.best_params_)

    mean_train_score = clf.cv_results_['mean_train_score']
    mean_test_score = clf.cv_results_['mean_test_score']
    print('mean train accuracy: {}'.format(mean_train_score))
    print('mean test accuracy: {}'.format(mean_test_score))

    clf_score = clf.score(test_data[:500], test_labels[:500])
    print('polynomial kernel score: {:.4f}'.format(clf_score))
    print()


initialize_data()
linear_svm()
rbf_kernel()
polynomial_kernel()
combined_computation()
