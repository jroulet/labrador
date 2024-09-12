"""
Integration test of the modules for generating data and training, namely:
    * generate_parameters
    * simulation
    * compression
    * rescaling
    * training
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

# pylint: disable=wrong-import-position
import tempfile
import textwrap
import tracemalloc
from unittest import TestCase, main
import numpy as np

from cogwheel_machine import (compression,
                              generate_parameters,
                              rescaling,
                              simulation,
                              training,
                              utils)
# pylint: enable=wrong-import-position


class TrainingDataTestCase(TestCase):
    """Class to test simulations and training."""
    def test_make_training_data(self):
        """
        Generate a small amount of training data in a temporary
        directory, and train a model on the CPU for a few epochs.
        """
        tracemalloc.start()

        with tempfile.TemporaryDirectory() as parentdir:
            # Generate training data
            rundir = utils.setup_rundir(parentdir)
            generate_parameters.main(rundir)
            simulation.main(rundir)

            size, peak = tracemalloc.get_traced_memory()
            print(f'{size=}, {peak=}')

            compression.create_mask(rundir)
            compression.svd_compression(rundir)

            rescaling.rescale_parameters(rundir)
            self._assert_unrescale_undoes_rescale(rundir)

            print('Created these training data:')
            os.system(f'tree {parentdir}')

            self._assert_same_training_and_testing_files(rundir)

            # Train a model for a couple epochs on the CPU
            # - Default:
            extra_lines = textwrap.dedent('''\
                TRAIN_KWARGS.update(max_num_epochs=2,
                                    training_batch_size=10)
                DEVICE = 'cpu'
                ''')
            self._train_model(rundir, extra_lines)

            # - Embedding network:
            extra_lines += textwrap.dedent('''\
                EMBEDDING_LAYER_SIZES = [16, 8]
                ''')
            self._train_model(rundir, extra_lines)

    @staticmethod
    def _train_model(rundir, extra_lines=''):
        modeldir = utils.setup_modeldir(rundir)
        with open(modeldir/utils.MODEL_CONFIG_FILENAME, 'a',
                  encoding='utf-8') as file:
            file.write(extra_lines)
        training.main(modeldir)

    def _assert_same_training_and_testing_files(self, rundir):
        training_files = set(os.listdir(rundir/utils.TRAINING_DIR))
        test_files = set(os.listdir(rundir/utils.TEST_DIR))
        self.assertEqual(training_files, test_files)

    @staticmethod
    def _assert_unrescale_undoes_rescale(rundir):
        rescaler = rescaling.ParameterRescaler(rundir)
        datadir = rundir/utils.TRAINING_DIR
        mask = np.load(datadir/utils.MASK_FILENAME)
        compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]
        folded_sampled_params = np.load(
            datadir/utils.FOLDED_SAMPLED_PARAMS_FILENAME)[mask]
        rescaled_parameters = np.load(
            datadir/utils.RESCALED_PARAMETERS_FILENAME)
        unrescaled = rescaler.unrescale(compressed_data, rescaled_parameters)
        np.testing.assert_almost_equal(folded_sampled_params, unrescaled)



if __name__ == '__main__':
    main()
