"""Mutant Networks.

Describe "Mutant Networks" problem using the artificial library.

Examples:
    >>> net = MutantNetwork([23, -1, 321, -1, 421, 41, 0, 123, 321, 31, 31])
    >>> model = net.model_from_data()
    >>> model.predict(...)

"""
import copy
from enum import Enum

import artificial as art
import keras.backend as K
import tensorflow as tf
from keras import optimizers
from keras.engine import InputLayer
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.utils import check_random_state


class Codes(Enum):
    Conv2D = 1
    Dense = 2
    MaxPooling2D = 3
    AveragePooling2D = 4
    Flatten = 5


layers_map = {
    Codes.Conv2D: Conv2D,
    Codes.Dense: Dense,
    Codes.MaxPooling2D: MaxPooling2D,
    Codes.Flatten: Flatten,
}


class Architect:
    """Architect.

    Projects random network architectures.

    Parameters
    ----------
    min_kernels: int, minimum number of kernels for an arbitrary
                 convolutional layer.
    max_kernels: int, maximum number of kernels for an arbitrary
                 convolutional layer.
    min_conv_layers: int, minimum number of convolutional layers.
    max_conv_layers: int, maximum number of convolutional layers.
    min_units: int, minimum number of units for an arbitrary
                    dense layer.
    max_units: int, maximum number of units for an arbitrary
                    dense layer.
    min_dense_layers: int, minimum number of dense layers.
    max_dense_layers: int, maximum number of dense layers.
    random_state: RandomState instance.

    """

    VARIANCES_AVAILABLE = {
        'narrow_dense',
        'narrow_conv2d',
        'expand_dense',
        'expand_conv2d',
        'reduce_top_layer_dense',
        'reduce_top_layer_conv2d',
        'increase_top_layer_dense',
        'increase_top_layer_conv2d'
    }

    def __init__(self, min_kernels: int, max_kernels: int,
                 min_conv_layers: int, max_conv_layers: int,
                 min_units: int, max_units: int,
                 min_dense_layers: int, max_dense_layers: int,
                 random_state=None):
        self.min_kernels = min_kernels
        self.max_kernels = max_kernels
        self.min_conv_layers = min_conv_layers
        self.max_conv_layers = max_conv_layers
        self.min_units = min_units
        self.max_units = max_units
        self.min_dense_layers = min_dense_layers
        self.max_dense_layers = max_dense_layers

        self.random_state = check_random_state(random_state)

    def random_network(self):
        r = self.random_state

        architecture = {
            Codes.Conv2D: [
                self.random_layer(Codes.Conv2D)
                for _ in range(r.randint(self.min_conv_layers,
                                         self.max_conv_layers + 1))],
            Codes.Dense: [
                self.random_layer(Codes.Dense)
                for _ in range(r.randint(self.min_dense_layers,
                                         self.max_dense_layers + 1))],
        }

        return self.validate(architecture)

    def random_layer(self, layer_type: Codes, **kwargs) -> dict:
        layer = {'_type': layer_type}

        if layer_type is Codes.Dense:
            layer['_attributes'] = {
                'output_dim': self.random_state.randint(self.min_units,
                                                        self.max_units + 1),
                'activation': 'relu',
            }
        elif layer_type is Codes.Conv2D:
            layer['_attributes'] = {
                'nb_filter': self.random_state.randint(self.min_kernels,
                                                       self.max_kernels + 1),
                'nb_row': 3,
                'nb_col': 3,
                'activation': 'relu',
            }
        else:
            raise ValueError('unknown layer type %s' % layer_type)

        layer.update(kwargs)
        return layer

    def validate(self, architecture):
        if architecture is None:
            return None

        # Make sure it has increasing kernel counts.
        architecture[Codes.Conv2D].sort(
            key=lambda x: x['_attributes']['nb_filter'])

        # Make sure it doesn't contain multiple
        # directly stacked pooling layers.
        # layers = architecture[Codes.Conv2D]
        # for i in range(len(layers)):
        #     if (layers[i]['_type'] is Codes.MaxPooling2D and
        #         layers[i + 1]['_type'] is Codes.MaxPooling2D):
        #         layers.pop(i)

        return architecture

    def update(self, architecture, variance):
        architecture = getattr(self, variance)(copy.deepcopy(architecture))
        return self.validate(architecture)

    def _narrow(self, architecture, code, min_layers):
        layers = architecture[code]
        if len(layers) > min_layers:
            layers.pop()
            return architecture

    def narrow_dense(self, architecture):
        return self._narrow(architecture, Codes.Dense, self.min_dense_layers)

    def narrow_conv2d(self, architecture):
        return self._narrow(architecture, Codes.Conv2D, self.min_conv_layers)

    def _expand(self, architecture, code, max_layers):
        layers = architecture[code]
        if len(layers) < max_layers:
            layers.append(self.random_layer(code))
            return architecture

    def expand_dense(self, architecture):
        return self._expand(architecture, Codes.Dense, self.max_dense_layers)

    def expand_conv2d(self, architecture):
        return self._expand(architecture, Codes.Conv2D, self.max_conv_layers)

    def _reduce_top_layer(self, architecture, code, min_units):
        layers = architecture[code]
        if layers:
            layer = layers.pop()
            units_field = code is Codes.Dense and 'output_dim' or 'nb_filter'

            n_units = layer['_attributes'][units_field]
            if n_units > min_units:
                layer['_attributes'][units_field] = max(min_units,
                                                        int(n_units / 1.5))
                layers.append(layer)
                return architecture

    def reduce_top_layer_dense(self, architecture):
        return self._reduce_top_layer(architecture, Codes.Dense,
                                      self.min_units)

    def reduce_top_layer_conv2d(self, architecture):
        return self._reduce_top_layer(architecture, Codes.Conv2D,
                                      self.min_kernels)

    def _increase_top_layer(self, architecture, code, max_units):
        layers = architecture[code]
        if layers:
            layer = layers.pop()
            units_field = code is Codes.Dense and 'output_dim' or 'nb_filter'

            n_units = layer['_attributes'][units_field]
            if n_units < max_units:
                layer['_attributes'][units_field] = min(max_units,
                                                        int(1.5 * n_units))
                layers.append(layer)
                return architecture

    def increase_top_layer_dense(self, architecture):
        return self._increase_top_layer(architecture, Codes.Dense,
                                        self.max_units)

    def increase_top_layer_conv2d(self, architecture):
        return self._increase_top_layer(architecture, Codes.Conv2D,
                                        self.max_kernels)

    def combine(self, a, b):
        r = self.random_state
        c = {}

        for code in (Codes.Conv2D, Codes.Dense):
            cp_a, cp_b = (r.randint(len(a[code]) + 1),
                          r.randint(len(b[code]) + 1))

            c[code] = a[code][:cp_a] + b[code][cp_b:]

            # Adjust inconsistent cases.
            # Child network has less layers than minimum.
            min_layers = (code is Codes.Dense and self.min_dense_layers or
                          self.min_conv_layers)
            if len(c[code]) < min_layers:
                c[code] += [self.random_layer(code)
                            for _ in range(min_layers - len(c[code]))]

            # Child network has more layers than maximum.
            max_layers = (code is Codes.Dense and self.max_dense_layers or
                          self.max_conv_layers)
            if len(c[code]) > max_layers:
                for _ in range(len(c[code]) - max_layers):
                    c[code].pop()

        return self.validate(c)

    def mutate(self, architecture, factor, probability):
        r = self.random_state

        for code in (Codes.Conv2D, Codes.Dense):
            layers = architecture[code]

            if code is Codes.Dense:
                units_field = 'output_dim'
                min_units, max_units = self.min_units, self.max_units
            else:
                units_field = 'nb_filter'
                min_units, max_units = self.min_kernels, self.max_kernels

            for layer_id in range(len(layers)):
                if r.rand() < probability:
                    layer = layers[layer_id]
                    n_units = layer['_attributes'][units_field]
                    n_units = int(factor * r.randint(min_units, max_units + 1)
                                  + (1 - factor) * n_units)
                    layer['_attributes'][units_field] = n_units

        return architecture


class MutantNetwork(art.base.GeneticState):
    loss_ = validation_loss_ = None

    def validate(self):
        env = Environment.current()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as s:
            K.set_session(s)

            with tf.device(env.consts.device):
                estimator = self.model_from_data()
                results = estimator.fit_generator(
                    generator=env.dataset_,
                    samples_per_epoch=env.dataset_.N,
                    nb_epoch=env.consts.n_epochs,
                    validation_data=env.val_dataset_)
        del s

        self.validation_loss_, self.loss_ = min(
            zip(results.history['val_loss'],
                results.history['loss']))
        return self

    def h(self) -> float:
        a = Environment.current().architect_

        unit_density = (
            sum((layer['_attributes']['nb_filter'] - a.min_kernels) /
                (a.max_kernels - a.min_kernels)
                for layer in self.data[Codes.Conv2D]) +
            sum((layer['_attributes']['output_dim'] - a.min_units) /
                (a.max_units - a.min_units)
                for layer in self.data[Codes.Dense]))

        n_layers = sum(map(len, (self.data.values())))
        layer_density = n_layers / (a.max_conv_layers + a.max_dense_layers)

        self.validate()

        mi = Environment.current().consts.metrics_importance
        h = (mi['unit_density'] * unit_density +
             mi['layer_density'] * layer_density +
             mi['train_loss'] * self.loss_ +
             mi['validation_loss'] * self.validation_loss_)

        tf.logging.info('architecture {%s}:\n'
                        '|-loss: %f\n'
                        '|-validation loss: %f\n'
                        '|-h(): %f', self, self.loss_, self.validation_loss_,
                        h)
        return h

    def model_from_data(self, compiled=True) -> Sequential:
        env = Environment.current()
        m = Sequential([InputLayer(env.consts.input_shape)])

        for i, layer in enumerate(self.data[Codes.Conv2D]):
            m.add(layers_map[layer['_type']](**layer['_attributes']))

            if (i + 1) % 2 == 0:
                m.add(MaxPooling2D())

        # If there's at least one conv2d layer, flatten the signal.
        m.add(Flatten())

        for layer in self.data[Codes.Dense]:
            m.add(layers_map[layer['_type']](**layer['_attributes']))

        # We are only dealing with classification right now.
        m.add(Dense(env.consts.n_classes, activation='softmax'))

        if compiled:
            optimizer = (env.optimizer != 'default' and env.optimizer or
                         optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9,
                                        nesterov=True))
            m.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return m

    def cross(self, other):
        return self.__class__(Environment.current()
                              .architect_.combine(self.data, other.data))

    def mutate(self, factor, probability):
        return self.__class__(Environment.current().architect_
                              .mutate(self.data, factor, probability))

    @classmethod
    def random(cls):
        return cls(Environment.current().architect_.random_network())

    def __str__(self):
        consts = Environment.current().consts

        return str([(l['_type'].name, l['_attributes']['nb_filter'],
                     l['_attributes']['activation'])
                    for l in self.data[Codes.Conv2D]] +
                   [(Codes.Flatten.name,)] +
                   [(l['_type'].name,
                     l['_attributes']['output_dim'],
                     l['_attributes']['activation'])
                    for l in self.data[Codes.Dense]] +
                   [(Codes.Dense.name, consts.n_classes, 'softmax')])


class Environment(art.base.Environment):
    """Mutant Environment.

    Environment for the exploration of architecturally mutant networks.

    """
    state_class_ = MutantNetwork

    def __init__(self, consts, initial_state=None, architect=None,
                 optimizer='default'):
        super().__init__(initial_state=initial_state)
        self.consts = consts
        self.optimizer = optimizer
        self.architect = architect

        self.dataset_ = self.val_dataset_ = self.test_dataset_ = None
        self.architect_ = architect or Architect(**consts.architect_params)

    def update(self):
        for agent in self.agents:
            try:
                self.current_state = agent.perceive().act()
            finally:
                # If the user interrupted the search, at least check if
                # a solution candidate currently exists.
                # A bad solution is still better than nothing.
                self.current_state = (agent.search.solution_candidate_
                                      or self.current_state)
        return self


class Agent(art.agents.ResponderAgent):
    """Mutant Catalyst Agent.

    Agent responsible for finding the best solution candidate (network
    architecture) for the digits-genetic dataset.

    Parameters
    ----------
    variances_allowed: set of strings, s.t. every element of this set is also
        an element of VARIANCES_AVAILABLE. Only these operations
        are signed as allowed to be performed by the mutation agent.

        The default is the `VARIANCES_AVAILABLE` set itself.

    """

    def __init__(self, search, environment,
                 variances_allowed=Architect.VARIANCES_AVAILABLE,
                 search_params=None, random_state=None):
        super().__init__(search=search, environment=environment,
                         search_params=search_params,
                         random_state=random_state)
        self.variances_allowed = variances_allowed

    def predict(self, state):
        architect = self.environment.architect_
        children = []

        for variance in self.variances_allowed:
            architecture = architect.update(state.data, variance=variance)

            if architecture:
                child = state.__class__(architecture, action=variance,
                                        parent=state)
                children.append(child)
        return children
