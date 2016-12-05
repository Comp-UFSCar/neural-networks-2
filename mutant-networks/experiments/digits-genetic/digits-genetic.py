"""Training and Predicting Digits with Mutant Networks.

The networks mutate their architecture through genetic algorithms.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import logging

import artificial as art
import numpy as np
import tensorflow as tf
from artificial.utils.experiments import arg_parser, ExperimentSet, Experiment
from keras.preprocessing.image import ImageDataGenerator
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import mutant


class DigitsMutantEnvironment(mutant.Environment):
    def build(self):
        X, y = load_digits(n_class=self.consts.n_classes, return_X_y=True)
        g = ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            width_shift_range=.1,
            height_shift_range=.1,
            rotation_range=30,
        )

        X = X.astype('float32')
        X /= 255

        test_split = self.consts.test_split
        X, X_test, y, y_test = train_test_split(X, y, test_size=test_split)

        X = X.reshape((X.shape[0], 8, 8, 1))
        X_test = X_test.reshape((X_test.shape[0], 8, 8, 1))

        g.fit(X)

        self.dataset_ = g.flow(X, y)
        self.val_dataset_ = self.test_dataset_ = (X_test, y_test)

        return self


class MutateOverDigitsExperiment(Experiment):
    env_ = None

    def setup(self):
        consts = self.consts

        np.random.seed(consts.seed)

        # Create mutation environment.
        env = DigitsMutantEnvironment(consts, optimizer='adam')
        env.agents = [
            art.agents.ResponderAgent(
                search=art.searches.genetic.GeneticAlgorithm,
                search_params=consts.search_params,
                environment=env,
                random_state=consts.agent_seed)
        ]

        self.env_ = env

    def run(self):
        self.env_.live(n_cycles=1)
        answer = self.env_.current_state
        if answer:
            tf.logging.info('solution candidate\'s train and validation loss: '
                            '(%f, %f)', self.consts.n_epochs,
                            answer.loss_, answer.validation_loss_)


if __name__ == '__main__':
    print(__doc__, flush=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('tensorflow').propagate = False

    (ExperimentSet(MutateOverDigitsExperiment)
     .load_from_json(arg_parser.parse_args().constants)
     .run())
