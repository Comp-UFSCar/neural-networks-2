"""Problem 2.

Author: Lucas David -- <ld492@drexel.edu>

"""
import matplotlib.pyplot as plt
import numpy as np
from algorithms.linear_estimator import Perceptron
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat
from sklearn.metrics import accuracy_score

Axes3D


def a(X, y):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

    plt.grid()
    plt.tight_layout(0)
    plt.savefig('results/p2.a.png')


def b(X, y):
    clf = Perceptron(learning_rate=.1, n_epochs=200, random_state=0)

    clf.fit(X, y)
    s = clf.predict(X)
    print(y)
    print(s)
    print('score: %.2f' % accuracy_score(y, s))
    print('epochs needed: %i' % clf.n_epochs_)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(range(len(clf.loss_)), clf.loss_)

    ax = fig.add_subplot(122, projection='3d')

    xx, yy = np.meshgrid(range(10), range(10))

    normal = clf.W_
    z = (-normal[0] * xx - normal[1] * yy - clf.b_) * 1. / normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2, color='green')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

    plt.grid()
    plt.tight_layout(0)
    plt.savefig('results/p2b.png')


def c(X, y):
    clf = Perceptron(learning_rate=.0001, n_epochs=2, random_state=0)

    clf.fit(X, y)
    s = clf.predict(X)
    print(y)
    print(s)
    print('score: %.2f' % accuracy_score(y, s))
    print('epochs needed: %i' % clf.n_epochs_)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(range(len(clf.loss_)), clf.loss_)

    ax = fig.add_subplot(122, projection='3d')

    xx, yy = np.meshgrid(range(10), range(10))

    normal = clf.W_
    z = (-normal[0] * xx - normal[1] * yy - clf.b_) * 1. / normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2, color='green')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

    plt.grid()
    plt.tight_layout(0)
    plt.savefig('results/p2c.png')


def d(X, y):
    clf = Perceptron(learning_rate=.1, n_epochs=200, random_state=1)

    clf.fit(X, y)
    s = clf.predict(X)
    print(y)
    print(s)
    print('score: %.2f' % accuracy_score(y, s))
    print('epochs needed: %i' % clf.n_epochs_)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.plot(range(len(clf.loss_)), clf.loss_)

    ax = fig.add_subplot(122, projection='3d')

    xx, yy = np.meshgrid(range(10), range(10))

    normal = clf.W_
    z = (-normal[0] * xx - normal[1] * yy - clf.b_) * 1. / normal[2]
    ax.plot_surface(xx, yy, z, alpha=0.2, color='green')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

    plt.grid()
    plt.tight_layout(0)
    plt.savefig('results/p2d.png')


def main():
    print(__doc__)

    data = loadmat('./data/dados1.mat')
    X, y = data['x'], data['desejado'].flatten()
    # y[y == -1] = 0

    d(X, y)


if __name__ == '__main__':
    main()
