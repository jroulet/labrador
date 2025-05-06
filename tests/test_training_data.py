"""Integration test."""
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
                              injections,
                              postprocessing,
                              pp_plot,
                              rescaling,
                              simulation,
                              training,
                              unfolding,
                              utils,
                              weighting)
# pylint: enable=wrong-import-position


class IntegrationTestCase(TestCase):
    """Class to test simulations and training."""
    def test_make_training_data(self, parentdir=None):
        """
        Generate a small amount of training data in a temporary
        directory, and train a sbi on the CPU for a few epochs.

        Parameters
        ----------
        parentdir : os.PathLike (optional)
            Path where a new directory containing training data will be
            created. If not provided, a temporary directory will be used
            and the data will be lost.
        """
        if parentdir is None:
            with tempfile.TemporaryDirectory() as tmp_parentdir:
                self.integration_test(tmp_parentdir)
        else:
            self.integration_test(parentdir)

    def integration_test(self, parentdir):
        """
        Run cogwheel-machine as a pipeline, end-to-end.

        This function generates a small amount of training data, trains
        a rescaler for a few epochs, trains a simulation-based
        inference, trains an unfolding classifier.

        Parameters
        ----------
        parentdir : os.PathLike
            Directory where to put all the generated files. It will be
            created if it doesn't exist.
        """
        tracemalloc.start()
        # Generate training data
        rundir = utils.setup_rundir(parentdir)
        generate_parameters.main(rundir)
        simulation.main(rundir, processes=10)

        size, peak = tracemalloc.get_traced_memory()
        print(f'{size=}, {peak=}')

        compression.create_mask(rundir)
        compression.svd_compression(rundir)

        weighting.main(rundir)

        priordirs = utils.get_priordirs(rundir)
        for priordir in priordirs:
            rescalerdir = utils.setup_rescalerdir(priordir)
            rescaling.main(rescalerdir)
            self._assert_unrescale_undoes_rescale(rescalerdir)

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
            sbidir = self._train_sbi(rescalerdir, extra_lines)

            unfolderdir = self._train_unfolding_classifier(rescalerdir)

            print('Making pp-plot...')
            pp_plot.main(sbidir, n_data=10, n_processes=2)
            print('Done.')

            self._event_end_to_end(sbidir, unfolderdir)

        print('Created these files:')
        os.system(f'tree {parentdir}')

    @staticmethod
    def _train_sbi(rescalerdir, extra_lines=''):
        sbidir = utils.setup_sbidir(rescalerdir)

        print(f'Training sbi in {sbidir}...')
        with open(sbidir/utils.SBI_CONFIG_FILENAME, 'a',
                  encoding='utf-8') as file:
            file.write(extra_lines)
        training.main(sbidir)
        print('Done.')

        return sbidir

    @staticmethod
    def _train_unfolding_classifier(rescalerdir):
        unfolderdir = utils.setup_unfolderdir(rescalerdir)
        unfolding.main(unfolderdir)
        return unfolderdir

    @staticmethod
    def _event_end_to_end(sbidir, unfolderdir):
        posterior = postprocessing.Posterior.from_tree(sbidir, unfolderdir)

        _, compressed_data, transform = injections.generate_data_and_transform(
            rundir=sbidir.parents[2])

        samples, lnprob_standard = posterior.generate_samples_and_lnprob(
            100, compressed_data, transform)

        assert set(transform.standard_params) <= set(samples)
        assert set(transform.sampled_params) <= set(samples)
        assert len(lnprob_standard) == len(samples)

    def _assert_same_training_and_testing_files(self, rundir):
        training_files = set(os.listdir(rundir/utils.TRAINING_DIR))
        test_files = set(os.listdir(rundir/utils.TEST_DIR))
        self.assertEqual(training_files, test_files)

    def _assert_unrescale_undoes_rescale(self, rescalerdir):
        rescaler = rescaling.ParameterRescaler(rescalerdir)
        rescaled_datadir = rescalerdir/utils.TRAINING_DIR
        rundir = rescalerdir.parents[1]
        datadir = rundir/utils.TRAINING_DIR
        mask = np.load(datadir/utils.MASK_FILENAME)
        compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]
        with h5py.File(datadir/utils.FOLDED_SAMPLED_PARAMETERS_FILENAME, "r"
                      ) as h5file:
            folded_sampled_parameters = h5file["dataset"][mask]

        rescaled_parameters = np.load(
            rescaled_datadir/utils.RESCALED_PARAMETERS_FILENAME)
        unrescaled, lnj = rescaler.unrescale(
            compressed_data, rescaled_parameters)
        unrescaled = unrescaled.detach().cpu()
        np.testing.assert_almost_equal(folded_sampled_parameters, unrescaled)

        self.assertEqual(lnj.shape, rescaled_parameters.shape[:-1])


if __name__ == '__main__':
    main()
