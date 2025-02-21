"""
Integration test of the modules for generating data and training, i.e.:

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
import h5py

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
        directory, and train a sbi on the CPU for a few epochs.
        """
        tracemalloc.start()

        with tempfile.TemporaryDirectory() as parentdir:
            # Generate training data
            rundir = utils.setup_rundir(parentdir)
            rescalerdir = utils.setup_rescalerdir(rundir)
            generate_parameters.main(rundir)
            simulation.main(rundir)

            size, peak = tracemalloc.get_traced_memory()
            print(f'{size=}, {peak=}')

            compression.create_mask(rundir)
            compression.svd_compression(rundir)

            rescaling.main(rescalerdir)
            self._assert_unrescale_undoes_rescale(rescalerdir)

            print('Created these training data:')
            os.system(f'tree {parentdir}')

            self._assert_same_training_and_testing_files(rundir)
            self._assert_same_training_and_testing_files(rescalerdir)

            # Train sbi for a couple epochs on the CPU
            # - Default:
            extra_lines = textwrap.dedent('''\
                TRAIN_KWARGS.update(max_num_epochs=2,
                                    training_batch_size=10)
                DEVICE = 'cpu'
                ''')
            self._train_sbi(rescalerdir, extra_lines)

            # - Embedding network:
            extra_lines += textwrap.dedent('''\
                EMBEDDING_LAYER_SIZES = [16, 8]
                ''')
            self._train_sbi(rescalerdir, extra_lines)

    @staticmethod
    def _train_sbi(rescalerdir, extra_lines=''):
        sbidir = utils.setup_sbidir(rescalerdir)
        with open(sbidir/utils.SBI_CONFIG_FILENAME, 'a',
                  encoding='utf-8') as file:
            file.write(extra_lines)
        training.main(sbidir)

    def _assert_same_training_and_testing_files(self, rundir):
        training_files = set(os.listdir(rundir/utils.TRAINING_DIR))
        test_files = set(os.listdir(rundir/utils.TEST_DIR))
        self.assertEqual(training_files, test_files)

    @staticmethod
    def _assert_unrescale_undoes_rescale(rescalerdir):
        rescaler = rescaling.ParameterRescaler(rescalerdir)
        rescaled_datadir = rescalerdir/utils.TRAINING_DIR
        rundir = rescalerdir.resolve().parent
        datadir = rundir/utils.TRAINING_DIR
        mask = np.load(datadir/utils.MASK_FILENAME)
        compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]
        with h5py.File(datadir/utils.FOLDED_SAMPLED_PARAMETERS_FILENAME, "r"
                      ) as h5file:
            folded_sampled_parameters = h5file["dataset"][mask]

        rescaled_parameters = np.load(
            rescaled_datadir/utils.RESCALED_PARAMETERS_FILENAME)
        unrescaled = rescaler.unrescale(compressed_data,
                                        rescaled_parameters).detach().cpu()
        np.testing.assert_almost_equal(folded_sampled_parameters, unrescaled)



if __name__ == '__main__':
    main()
