"""Training and Predicting Digits with Mutant Networks.

The networks mutate their architecture through hill-climbing.

Author: Lucas David -- <ld492@drexel.edu>
Licence: MIT License 2016 (c)

"""
import logging
import time

import artificial as art
import mutant
import numpy as np
import tensorflow as tf
from artificial.utils.experiments import arg_parser, ExperimentSet, Experiment
from keras.preprocessing.image import ImageDataGenerator
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class DigitsMutantEnvironment(mutant.Environment):
    def build(self):
        tf.logging.info('building environment...')
        tf.logging.info('|-loading data...')

        X, y = load_digits(n_class=self.consts.n_classes, return_X_y=True)
        g = ImageDataGenerator(
            rescale=1.0 / 255.0,
            featurewise_center=True,
            featurewise_std_normalization=True,
            width_shift_range=.1,
            height_shift_range=.1,
            rotation_range=30)

        test_split = self.consts.test_split
        X, X_val, y, y_val = train_test_split(X, y, test_size=test_split)

        X = X.reshape((X.shape[0], 8, 8, 1))
        X_val = X_val.reshape((X_val.shape[0], 8, 8, 1))

        tf.logging.info('|-fitting image generator...')
        g.fit(X)

        tf.logging.info('|-defining data sets...')
        self.dataset_ = g.flow(X, y)
        self.val_dataset_ = (X_val, y_val)

        tf.logging.info('building complete')
        return self


class HillClimbOverDigitsExperiment(Experiment):
    env_ = None

    def setup(self):
        consts = self.consts
        np.random.seed(consts.seed)

        # Create mutation environment.
        env = DigitsMutantEnvironment(consts=consts)

        env.agents = [
            mutant.Agent(search=art.searches.local.HillClimbing,
                         search_params=consts.search_params,
                         environment=env,
                         random_state=consts.agent_seed)
        ]

        self.env_ = env

    def run(self):
        self.env_.live(n_cycles=1)
        answer = self.env_.current_state

        if answer:
            time.sleep(.1)
            print('best model:', answer)
            print('|-loss after %i epochs: %f'
                  % (self.consts.n_epochs, answer.loss_))
            print('|-validation-loss after %i epochs: %f'
                  % (self.consts.n_epochs, answer.validation_loss_))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('tensorflow').propagate = False

    print(__doc__, flush=True)
    (ExperimentSet(HillClimbOverDigitsExperiment)
     .load_from_json(arg_parser.parse_args().constants)
     .run())
