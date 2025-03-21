"""
Integration test of the modules for generating data and training, i.e.:

    * generate_parameters
    * simulation
    * compression
    * rescaling
    * training
    * unfolding

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
                              postprocessing,
                              rescaling,
                              simulation,
                              training,
                              unfolding,
                              utils)
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
        rescalerdir = utils.setup_rescalerdir(rundir)
        generate_parameters.main(rundir)
        simulation.main(rundir)

        size, peak = tracemalloc.get_traced_memory()
        print(f'{size=}, {peak=}')

        compression.create_mask(rundir)
        compression.svd_compression(rundir)

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

        self._event_end_to_end(sbidir, unfolderdir)

        print('Created these files:')
        os.system(f'tree {parentdir}')

    @staticmethod
    def _train_sbi(rescalerdir, extra_lines=''):
        sbidir = utils.setup_sbidir(rescalerdir)
        with open(sbidir/utils.SBI_CONFIG_FILENAME, 'a',
                  encoding='utf-8') as file:
            file.write(extra_lines)
        training.main(sbidir)
        return sbidir

    @staticmethod
    def _train_unfolding_classifier(rescalerdir):
        unfolderdir = utils.setup_unfolderdir(rescalerdir)
        unfolding.main(unfolderdir)
        return unfolderdir

    @staticmethod
    def _event_end_to_end(sbidir, unfolderdir):
        rescalerdir, rundir = sbidir.parents[:2]
        preprocessed_data, transform \
            = generate_preprocessed_data_and_transform(rundir)

        compressed_data = compression.compress_data(
            rundir,
            preprocessed_data['heterodyned_data'],
            preprocessed_data['processed_coef'])

        sbi_posterior = training.load_posterior(sbidir)

        unfolding_classifier = unfolding.UnfoldingClassifier(unfolderdir)
        parameter_rescaler = rescaling.ParameterRescaler(rescalerdir)
        postprocessor = postprocessing.PostProcessor(
            unfolding_classifier, parameter_rescaler, transform)

        rescaled_parameters = sbi_posterior.sample([100], x=compressed_data)
        samples = postprocessor.postprocess_samples(compressed_data,
                                                    rescaled_parameters)
        assert set(transform.standard_params) <= set(samples)

    def _assert_same_training_and_testing_files(self, rundir):
        training_files = set(os.listdir(rundir/utils.TRAINING_DIR))
        test_files = set(os.listdir(rundir/utils.TEST_DIR))
        self.assertEqual(training_files, test_files)

    @staticmethod
    def _assert_unrescale_undoes_rescale(rescalerdir):
        rescaler = rescaling.ParameterRescaler(rescalerdir)
        rescaled_datadir = rescalerdir/utils.TRAINING_DIR
        rundir = rescalerdir.parent
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


def generate_preprocessed_data_and_transform(rundir):
    """
    Return preprocessed data and transform for a random simulated event.

    Parameters
    ----------
    rundir : os.PathLike
        Path to run directory.

    Returns
    -------
    preprocessed_data : dict
        Contains keys 'heterodyned_data', 'processed_coef', etc.

    transform : cogwheel_machine.transform.TransformMixin
        Instance of the transform class that corresponds to these data.
    """
    data_config = utils.load_data_config(rundir)
    prior = data_config.PRIOR_CLASS(**data_config.PRIOR_KWARGS)
    parameters = prior.generate_random_samples(1).iloc[0]

    simulator, data_preprocessor, transform_class \
        = simulation.setup_simulator(rundir)

    simulated_input = simulator.generate_data_and_reference_waveform(
        parameters)
    preprocessed_data, transform_kwargs \
        = data_preprocessor.preprocess_data(**simulated_input)

    transform = transform_class(**transform_kwargs)
    return preprocessed_data, transform


if __name__ == '__main__':
    main()
