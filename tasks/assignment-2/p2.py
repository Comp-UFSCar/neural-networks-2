"""Problem 2.

(a): Describe the data set, printing its balance.
(b): Propose a bi-partition for the training data, showing that the classes
     will maintain their original balance.
(c): Select attributes based on their associated Pearson's covariance.
(d): Train an extreme learning machine.
(e): Train a Fisher's Discriminant model.
(f): Train an SVM.

Author: Lucas David -- <ld492@drexel.edu>
License: MIT (C) 2011

"""

import numpy as np
from sklearn import svm, preprocessing, model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

FEATURES = ['age', 'workclass', 'fnlwgt', 'education', 'educationnum',
            'maritalstatus', 'occupation', 'relationship', 'race', 'sex',
            'capitalgain', 'capitalloss', 'hoursperweek', 'nativecountry',
            'income']

CATEGORICAL_FEATURES = ['workclass', 'education', 'maritalstatus',
                        'occupation', 'relationship', 'race', 'sex',
                        'nativecountry', 'income']

TARGET_FEATURE = 'income'


def load_data(ds_file, encoders=None):
    class Dataset:
        def __init__(self, data, target, encoders):
            self.data = data
            self.target = target
            self.encoders = encoders

    data = np.genfromtxt(ds_file, delimiter=',', dtype=None, names=FEATURES,
                         autostrip=True)
    y = None
    encoded = []
    encoders = encoders or {}
    for f in FEATURES:
        x = data[f]

        if f in CATEGORICAL_FEATURES:
            x = x.astype(np.str)

            if f in encoders:
                # Testing, use the fitted encoders.
                enc = encoders[f]
                x = enc.transform(x).astype(np.float)
            else:
                # Training, no encoder was defined yet.
                enc = preprocessing.LabelEncoder()
                encoders[f] = enc
                x = enc.fit_transform(x).astype(np.float)

            try:
                unknown_code = list(enc.classes_).index('?')
                x[x == unknown_code] = np.nan
            except ValueError:
                # Awesome. All values are present!
                pass

        if f == TARGET_FEATURE:
            y = x
        else:
            encoded.append(x)

    return Dataset(np.array(encoded, dtype=np.float).T, y, encoders)


def a_describe_dataset(train, test):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    Axes3D

    print('---\nX-train shape:', train.data.shape)
    print('X-test shape:', test.data.shape)
    print('train balance: %.4f' % (train.target == 0).mean())
    print('test balance: %.4f' % (test.target == 0).mean())

    def plot(X, y, name='X'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)

        plt.grid()
        plt.tight_layout(0)
        plt.savefig('results/%s.png' % name)

    # Plot data sets.
    p = Pipeline([
        ('i', preprocessing.Imputer(strategy='median')),
        ('p', PCA(n_components=3))])
    plot(p.fit_transform(train.data), train.target, 'train-data')
    plot(p.transform(test.data), test.target, 'test-data')


def b_propose_division(train, test):
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(train.data, train.target,
                                                      test_size=.3)

    # Print how balanced the validation partition is.
    # This number should be very close to the original balance in train.
    print('validation balance: %.4f' % (y_val == 0).mean())


def c_select_attributes(train, test):
    from scipy.stats import pearsonr

    N_ATTRIBUTES_SELECTED = 10

    print('imputing missing values...')
    X = train.data

    # Deal with unknowns by substituting them with their
    # most frequently observed value.
    X = preprocessing.Imputer(strategy='median').fit_transform(X)

    print('computing correlations...')
    correlations = [(tag, *pearsonr(X[:, i], train.target))
                    for i, tag in enumerate(FEATURES[:-1])]
    correlations.sort(key=lambda x: -abs(x[1]))

    tags, correlations = (list(tag for tag, _, _ in correlations),
                          np.array(list(corr for _, corr, _ in correlations)))

    print('correlations (feature, target):\n', list(zip(tags, correlations)))

    corr_rates = np.abs(correlations) / np.abs(correlations).sum()
    print('correlations explained:\n', list(zip(tags, corr_rates)))

    selected_corr_rates = corr_rates[:N_ATTRIBUTES_SELECTED]
    print('%i attributes consist of %.2f%% of all correlations.'
          % (N_ATTRIBUTES_SELECTED, selected_corr_rates.sum()))


def d_extreme_machine(train, test):
    from keras import layers, callbacks, models, regularizers

    EXECUTIONS_PARAMS = [{'inner_units': 1024, 'reg_lambda': 0.1},
                         {'inner_units': 1024, 'reg_lambda': 0.5},
                         {'inner_units': 1024, 'reg_lambda': 0.9}]

    # Missing values imputing, one-hot encoding and data whitening.
    X, y = train.data, train.target
    categorical_features = np.in1d(FEATURES[:-1], CATEGORICAL_FEATURES,
                                   assume_unique=True)
    flow = Pipeline([
        ('imput', preprocessing.Imputer(strategy='median')),
        ('one_hot', preprocessing.OneHotEncoder(
            categorical_features=categorical_features, sparse=False)),
        ('pca', PCA(n_components=10, whiten=True)),
    ])

    print('training PCA and transforming train data...')
    W_train = flow.fit_transform(X)

    # Print info about the maintained variance of the data.
    pca = flow.named_steps['pca']
    print('%i components explain %.2f of the variance: %s'
          % (pca.n_components_, pca.explained_variance_ratio_.sum(),
             pca.explained_variance_ratio_))

    for params in EXECUTIONS_PARAMS:
        LOG_NAME = '-'.join('%s:%s' % (k, str(v)) for k, v in params.items())

        em = models.Sequential([
            layers.InputLayer(input_shape=[pca.n_components_]),
            layers.Dense(params['inner_units'], activation='linear',
                         init='uniform', trainable=False),
            layers.Dense(2, activation='softmax',
                         W_regularizer=regularizers.l2(params['reg_lambda']))
        ])

        print('training extreme learning machine...')
        em.compile('SGD', 'sparse_categorical_crossentropy')
        em.fit(W_train, train.target, batch_size=1024, nb_epoch=1000,
               validation_split=.3,
               callbacks=[
                   callbacks.EarlyStopping(patience=5),
                   callbacks.TensorBoard('./results/logs/%s'
                                         % LOG_NAME),
               ])

        Z = em.predict(W_train, batch_size=1024)
        p = np.argmax(Z, axis=1)
        print('training score: %.2f' % accuracy_score(train.target, p))

        Z = em.predict(flow.transform(test.data), batch_size=1024)
        p = np.argmax(Z, axis=1)
        print('test score: %.2f' % accuracy_score(test.target, p))


def e_fisher(train, test):
    def build_model(X, y):
        X_0, X_1 = X[y == 0], X[y == 1]
        u0, u1 = np.mean(X_0, axis=0), np.mean(X_1, axis=0)

        Sw = (X_0 - u0).T.dot(X_0 - u0) + (X_1 - u1).T.dot(X_1 - u1)
        W = np.linalg.inv(Sw).dot(u1 - u0)
        return W, (W.T.dot(u0) + W.T.dot(u1)) / 2

    # Missing values imputing, one-hot encoding and data whitening.
    categorical_features = np.in1d(FEATURES[:-1], CATEGORICAL_FEATURES,
                                   assume_unique=True)
    flow = Pipeline([
        ('imput', preprocessing.Imputer(strategy='median')),
        ('one_hot', preprocessing.OneHotEncoder(
            categorical_features=categorical_features, sparse=False)),
        ('pca', PCA(n_components=10, whiten=True)),
    ])

    print('training PCA and transforming train data...')
    X_train = flow.fit_transform(train.data)

    W, b = build_model(X_train, train.target)

    p = (X_train.dot(W) - b >= 0).astype(np.float)
    print('training score: %.2f' % accuracy_score(train.target, p))
    X_test = flow.transform(test.data)
    p = (X_test.dot(W) - b >= 0).astype(np.float)
    print('test score: %.2f' % accuracy_score(test.target, p))


def f_svm(train, test):
    categorical_features = np.in1d(FEATURES[:-1], CATEGORICAL_FEATURES,
                                   assume_unique=True)

    flow = Pipeline([
        ('imput', preprocessing.Imputer(strategy='median')),
        ('one_hot', preprocessing.OneHotEncoder(
            categorical_features=categorical_features, sparse=False)),
        ('pca', PCA(n_components=10, whiten=True)),
        ('svm', svm.SVC(class_weight='balanced')),
    ])

    grid = model_selection.GridSearchCV(
        flow, {'svm__C': [.01, .1, 1, 10, 100],
               'svm__kernel': ['linear', 'rbf']},
        n_jobs=2, verbose=2).fit(train.data, train.target)

    print('best params: %s' % grid.best_params_)
    print('best val. score (3-fold): %.2f' % grid.best_score_)
    print('train accuracy: %.2f' % grid.score(train.data, train.target))
    print('test accuracy: %.2f' % grid.score(test.data, test.target))

    return grid


def g_deep_networks(train, test):
    import tensorflow as tf
    from keras import layers, callbacks, models, backend as K

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.Session(config=config))

    # Missing values imputing, one-hot encoding and data whitening.
    categorical_features = np.in1d(FEATURES[:-1], CATEGORICAL_FEATURES,
                                   assume_unique=True)
    flow = Pipeline([
        ('imput', preprocessing.Imputer(strategy='median')),
        ('one_hot', preprocessing.OneHotEncoder(
            categorical_features=categorical_features, sparse=False)),
        ('pca', PCA(n_components=10, whiten=True)),
    ])

    with tf.device('/cpu:0'):
        em = models.Sequential([
            layers.InputLayer(input_shape=[10]),

            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(.5),

            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(.5),

            layers.Dense(2),
            layers.BatchNormalization(),
            layers.Activation('softmax')
        ])

        em.compile('adam', 'sparse_categorical_crossentropy')

        try:
            em.fit(flow.fit_transform(train.data), train.target,
                   batch_size=128, nb_epoch=1000, validation_split=.3,
                   callbacks=[
                       callbacks.EarlyStopping(patience=20),
                       callbacks.TensorBoard('./results/logs/dense-'
                                             'w-dropout-and-batch-norm'),
                   ])
        except KeyboardInterrupt:
            print('training interrupted.')

        p = np.argmax(em.predict(flow.transform(test.data), batch_size=1024), 1)
        print('test score: %.2f' % accuracy_score(test.target, p))


def main():
    print(__doc__)

    train = load_data('./data/adult/adult.data')
    test = load_data('./data/adult/adult.test', train.encoders)

    a_describe_dataset(train, test)
    # b_propose_division(train, test)
    # c_select_attributes(train, test)
    # d_extreme_machine(train, test)
    # e_fisher(train, test)
    # f_svm(train, test)
    # g_deep_networks(train, test)


if __name__ == '__main__':
    main()
