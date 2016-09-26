"""Problem 3.

Author: Lucas David -- <ld492@drexel.edu>

"""

import multiprocessing

from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
from sklearn.pipeline import Pipeline

from algorithms import ExtremeMachine

Axes3D

N_JOBS = multiprocessing.cpu_count()


def train(clf, params, X, y):
    grid = GridSearchCV(clf, params, n_jobs=N_JOBS)
    grid.fit(X, y)

    print('grid parameters: %s' % params)
    print('best parameters: %s' % grid.best_params_)
    print('best estimator\'s score in testing fold: %.2f' % grid.best_score_)

    evaluate(grid.best_estimator_, X, y)

    return grid


def evaluate(machine, X, y):
    print('score: %.2f' % machine.score(X, y))


def a(X, y):
    machine = MLPRegressor()
    params = {
        'hidden_layer_sizes': [(100,), (200,), (1024,)],
        'learning_rate_init': [.001],
        'max_iter': [100, 200, 300],
    }

    return train(machine, params, X, y)


def b(X, y):
    machine = Pipeline([('ex', ExtremeMachine()), ('rg', MLPRegressor())])
    params = {
        'ex__n_features': [32, 64, 128, 256, 512, 1024],
        'ex__random_state': [0],
        'rg__hidden_layer_sizes': [(100,), (200,), (1024,), (2048,)],
        'rg__random_state': [1],
        'rg__max_iter': [200],
    }

    return train(machine, params, X, y)


def c(X, y, X_test, y_test):
    evaluate(a(X, y).best_estimator_, X_test, y_test)
    evaluate(b(X, y).best_estimator_, X_test, y_test)


def main():
    print(__doc__)

    data = loadmat('./data/dados2.mat')
    X = data['ponto'][:, :2]
    y = data['ponto'][:, 2].flatten()

    data = loadmat('./data/dados3.mat')
    X_test = data['ponto'][:, :2]
    y_test = data['ponto'][:, 2].flatten()

    print('shapes: ', X.shape, y.shape)
    c(X, y, X_test, y_test)


if __name__ == '__main__':
    main()
