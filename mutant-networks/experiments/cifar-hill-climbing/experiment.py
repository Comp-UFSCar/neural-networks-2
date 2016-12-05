"""Training and Predicting Cifar10 with Mutant Networks.

The networks mutate their architecture using genetic algorithms.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import logging

import artificial as art
import numpy as np
import tensorflow as tf
from artificial.utils.experiments import arg_parser, ExperimentSet, Experiment
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

import mutant


class Cifar10MutantEnvironment(mutant.Environment):
    def build(self):
        tf.logging.info('building environment...')
        tf.logging.info('|-loading data...')
        (X, y), (X_test, y_test) = cifar10.load_data()

        X = X.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        g = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

        tf.logging.info('|-fitting image generator...')
        g.fit(X)

        tf.logging.info('|-defining data sets...')
        self.dataset_ = g.flow(X, y, batch_size=self.consts.batch_size,
                               shuffle=self.consts.shuffle_train)
        self.test_dataset_ = self.val_dataset_ = (X_test, y_test)
        tf.logging.info('building complete')
        return self


class ClimbOverCifar10Experiment(Experiment):
    env_ = None

    def setup(self):
        consts = self.consts

        # Settings for logging.
        verbosity_level = logging.INFO if consts.verbose else logging.WARNING
        for m in ('artificial', 'tensorflow', 'connoisseur'):
            logger = logging.getLogger(m)
            logger.setLevel(verbosity_level)
            logger.addHandler(logging.FileHandler(consts.log_file))

        np.random.seed(consts.seed)

        # Create mutation environment.
        e = Cifar10MutantEnvironment(optimizer='adam', consts=consts)

        e.agents = [
            mutant.Agent(search=art.searches.local.HillClimbing,
                         environment=e,
                         **consts.agent_params)
        ]

        initial_architecture = e.architect_.validate({
            mutant.Codes.Conv2D: [
                e.architect_.random_layer(mutant.Codes.Conv2D)
                for _ in range(4)
                ],
            mutant.Codes.Dense: [
                e.architect_.random_layer(mutant.Codes.Dense)
                for _ in range(2)
                ],
        })
        initial_state = mutant.MutantNetwork(initial_architecture)
        e.current_state = e.initial_state = initial_state

        self.env_ = e

    def run(self):
        try:
            self.env_.live(n_cycles=1)
        finally:
            answer = self.env_.current_state
            if answer:
                tf.logging.info('train and validation loss after %i epochs: '
                                '(%s, %s)', self.consts.n_epochs,
                                answer.loss_, answer.validation_loss_)


if __name__ == '__main__':
    print(__doc__, flush=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('tensorflow').propagate = False

    (ExperimentSet(ClimbOverCifar10Experiment)
     .load_from_json(arg_parser.parse_args().constants)
     .run())
