import numpy as np
import pandas as pd
from sklearn import svm

train_data = np.array([])
test_data = np.array([])
train_labels = np.array([])
test_labels = np.array([])
C = 1.0


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
    svc = svm.SVC(kernel='linear', C=C).fit(train_data[:100], train_labels[:100])
    score = svc.score(test_data[:100], test_labels[:100])
    print('linear svm score: {:.4f}'.format(score))


def rbf_kernel():
    gamma = 0.7
    rbf_svc = svm.SVC(kernel='rbf', gamma=gamma, C=C).fit(train_data[:100], train_labels[:100])
    rbf_svc_score = rbf_svc.score(test_data[:100], test_labels[:100])
    print('rbf kernel score: {:.4f}'.format(rbf_svc_score))


def polynomial_kernel():
    degree = 3
    poly_svc = svm.SVC(kernel='poly', degree=degree, C=C).fit(train_data[:100], train_labels[:100])
    poly_svc_score = poly_svc.score(test_data[:100], test_labels[:100])
    print('polynomial kernel score: {:.4f}'.format(poly_svc_score))


initialize_data()
linear_svm()
rbf_kernel()
polynomial_kernel()

