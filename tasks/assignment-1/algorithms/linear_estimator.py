import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import NotFittedError


def quadratic_loss(y, s):
    return np.average((y - s) ** 2 / 2)


class Perceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=.1, n_epochs=200, random_state=None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.W_ = self.b_ = self.loss_ = None
        self.random_state = random_state

    def feed_forward(self, X):
        W, b = self.W_, self.b_
        return X.dot(W) + b

    def fit(self, X, y, **fit_params):
        batch_size, features = X.shape

        # Random initiate.
        r = check_random_state(self.random_state)
        W, b = r.randn(features, 1), r.randn()
        self.W_, self.b_ = W, b
        self.loss_ = []

        for epoch in range(self.n_epochs):
            loss = 0

            for i, x in enumerate(X):
                s = self.predict(x)
                loss += quadratic_loss(y[i], s)

                d = y[i] - s
                dW = d * x.reshape(-1, 1)

                W += self.learning_rate * dW
                b -= self.learning_rate * np.average(d)

            self.loss_.append(loss / X.shape[0])

            if loss == 0:
                # Shortcut. We already got every sample correctly.
                break

        self.n_epochs_ = epoch
        return self

    def predict(self, X):
        if self.W_ is None:
            raise NotFittedError

        return np.sign(self.feed_forward(X)).flatten().astype(int)
