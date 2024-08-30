"""
Integration test of the modules for generating training data, namely:
    * generate_parameters
    * simulation
    * compression
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import tempfile
import textwrap
from unittest import TestCase, main

from cogwheel_machine import (compression,
                              generate_parameters,
                              simulation,
                              training,
                              utils)


class TrainingDataTestCase(TestCase):
    """Class to test simulations and training."""
    def test_make_training_data(self):
        """
        Generate a small amount of training data in a temporary
        directory, and train a model on the CPU for a few epochs.
        """
        with tempfile.TemporaryDirectory() as parentdir:
            # Generate training data
            rundir = utils.setup_rundir(parentdir)
            generate_parameters.main(rundir)
            simulation.main(rundir)
            compression.create_mask(rundir)
            compression.svd_compression(rundir)
            print('Created these training data:')
            os.system(f'tree {parentdir}')

            # Train a few models with different settings
            # - Default:
            self._train_model(rundir)

            # - Embedding network:
            extra_lines = textwrap.dedent('''\
                EMBEDDING_LAYER_SIZES = [16, 8]
                ''')
            self._train_model(rundir, extra_lines)

    @staticmethod
    def _train_model(rundir, extra_lines=''):
        extra_lines += textwrap.dedent('''\
            TRAIN_KWARGS.update(max_num_epochs=2,
                                training_batch_size=10)
            DEVICE = 'cpu'
            ''')
         # Train a model for a couple epochs on the CPU
        modeldir = utils.setup_modeldir(rundir)
        with open(modeldir/utils.MODEL_CONFIG_FILENAME, 'a') as file:
            file.write(extra_lines)
        training.main(modeldir)


if __name__ == '__main__':
    main()
