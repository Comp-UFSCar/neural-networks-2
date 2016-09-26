"""Problem 4.

Author: Lucas David -- <ld492@drexel.edu>

"""

import multiprocessing

from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, train_test_split

from algorithms import RBFRegressor

Axes3D

N_JOBS = multiprocessing.cpu_count() - 2
N_CLUSTERS = 50
N_FEATURES = 2


def train(clf, params, X, y):
    grid = GridSearchCV(clf, params, n_jobs=N_JOBS)
    grid.fit(X, y)

    print('grid parameters: %s' % params)
    print('best parameters: %s' % grid.best_params_)
    print('best estimator\'s score in validation fold: %.2f' % grid.best_score_)

    evaluate(grid.best_estimator_, X, y)
    return grid


def evaluate(machine, X, y):
    print('score: %.2f' % machine.score(X, y))


def a(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8,
                                                        random_state=0)

    clusterizer = KMeans(n_clusters=N_CLUSTERS, n_jobs=N_JOBS,
                         random_state=0)
    clusterizer.fit(X_train, y_train)

    print('centers: ', clusterizer.cluster_centers_)

    machine = RBFRegressor(n_features=N_FEATURES,
                           centers=clusterizer.cluster_centers_,
                           random_state=1)
    machine.fit(X_train, y_train)

    print('training', end=' ')
    evaluate(machine, X_train, y_train)

    print('validation', end=' ')
    evaluate(machine, X_test, y_test)

    return machine


def main():
    print(__doc__)

    data = loadmat('./data/dados_map.mat')
    X = data['dados_rbf'][:, :2]
    y = data['dados_rbf'][:, 2].flatten()

    print('shapes: ', X.shape, y.shape)
    a(X, y)


if __name__ == '__main__':
    main()
