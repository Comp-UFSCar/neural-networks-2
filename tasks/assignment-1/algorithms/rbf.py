import numpy as np
from scipy.linalg import norm, pinv
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state


class RBFRegressor(BaseEstimator, RegressorMixin):
    """RBF Network.

    Based on 'Radial Basis Function (RBF) Network for Python', by Thomas
    RuckstieB. Available at: [http://www.rueckstiess.net/research/
    snippets/show/72d2363e]

    """

    def __init__(self, n_features, centers=8, random_state=None):
        self.n_dimensions = n_features
        self.random_state = check_random_state(random_state)

        if isinstance(centers, int):
            self.n_centers = centers
            self.centers = self.random_state.uniform(-1, 1,
                                                     (centers, n_features))
        elif isinstance(centers, (list, tuple, set, np.ndarray)):
            self.n_centers = len(centers)
            self.centers = centers
        else:
            raise ValueError('Illegal value for centers: %s (%s)'
                             % (centers, type(centers)))

        self.beta_ = 8
        self.W_ = self.random_state.randn(self.n_centers, n_features)

    def _basis_function(self, c, d):
        return np.exp(-self.beta_ * norm(c - d) ** 2)

    def feed_forward(self, X):
        # calculate activations of RBFs
        G = np.zeros((X.shape[0], self.n_centers), float)
        for i, x in enumerate(X):
            for ci, c in enumerate(self.centers):
                G[i, ci] = self._basis_function(c, x)
        return G

    def fit(self, X, y):
        r = check_random_state(self.random_state)
        # choose r center vectors from training set
        rnd_idx = r.permutation(X.shape[0])[:self.n_centers]
        self.centers = [X[i, :] for i in rnd_idx]

        G = self.feed_forward(X)

        # calculate output weights (pseudo-inverse).
        self.W_ = np.dot(pinv(G), y)

        return self

    def predict(self, X):
        return np.dot(self.feed_forward(X), self.W_).flatten()
