"""Problem 1.

Author: Lucas David -- <ld492@drexel.edu>

"""
import numpy as np
from sklearn.svm import LinearSVR
import matplotlib.pyplot as plt


def quadratic_loss(model, X, y):
    return ((model.predict(X) - y) ** 2 / X.shape[0] / 2).sum()


def main():
    print(__doc__)

    X = np.array([0, 1, 2, 5, 7, 9]).reshape((6, 1))
    y = np.array([0, 2, 3, 8, 7, 10])

    print('X: %s, y: %s' % (X.flatten(), y))
    model = LinearSVR()
    model.fit(X, y)

    a, b = model.coef_, model.intercept_

    print('a: %.2f, b: %.2f' % (a, b))
    print('loss: %.2f' % quadratic_loss(model, X, y))

    plt.scatter(X, y, c=y, s=200)
    plt.plot(np.arange(10), a * np.arange(10) + b,
             c='green', lw=10, alpha=.4)

    plt.grid()
    plt.tight_layout()
    plt.savefig('results/p1.png')


if __name__ == '__main__':
    main()
