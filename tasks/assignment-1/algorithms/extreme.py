import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state


class ExtremeMachine(BaseEstimator, TransformerMixin):
    def __init__(self, n_features='auto', random_state=None):
        self.n_features = n_features
        self.random_state = random_state

        self.n_features_ = self.W_ = None

    def define_output_feature_space(self, n_input_features):
        if self.n_features_ is not None:
            return self.n_features_

        if self.n_features == 'auto':
            self.n_features_ = 128 * n_input_features
        elif isinstance(self.n_features, int):
            self.n_features_ = self.n_features
        elif isinstance(self.n_features, float):
            self.n_features_ = self.n_features * n_input_features

        return self.n_features_

    def fit(self, X, y=None, **fit_params):
        n_input = X.shape[1]
        n_output = self.define_output_feature_space(n_input)

        random = check_random_state(self.random_state)

        self.W_ = random.randn(n_input, n_output)
        return self

    def transform(self, X):
        return np.dot(X, self.W_)
