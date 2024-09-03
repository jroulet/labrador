"""
Algorithms for compressing the preprocessed data, and for defining a
mask to select a subset of the simulations.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import sklearn.preprocessing

import cogwheel.utils

import cogwheel.utils

from . import utils


def create_mask(rundir):
    """
    Create and save masks for the training and test data, specifying
    which simulations satisfy the cuts per ``config.MASK_CONDITIONS``.

    The mask is a boolean array of shape (n_simulations,) encoding
    which simulations satisfy the cuts.

    Parameters
    ----------
    rundir: os.PathLike
        Path to run directory.
    """
    rundir = Path(rundir)
    utils.check_version(rundir)

    config = utils.load_data_config(rundir)

    for datadir in rundir/utils.TRAINING_DIR, rundir/utils.TEST_DIR:
        summary = utils.get_summary(datadir, apply_mask=False)

        mask = np.full(len(summary), True)
        for par, logic, value in config.MASK_CONDITIONS:
            mask &= logic(summary[par], value)

        np.save(datadir/utils.MASK_FILENAME, mask)


def _get_data(datadir, data_getter):
    compressed_heterodyned_data = data_getter(datadir)

    with np.load(datadir/utils.PREPROCESSED_DATA_FILENAME) as file:
        processed_coef = file['processed_coef']

    return np.concatenate([compressed_heterodyned_data, processed_coef],
                          axis=1)


def _save_compressed_data(unscaled_data, scaler, datadir):
    compressed_data = scaler.transform(unscaled_data).astype(np.float32)
    np.save(datadir/utils.COMPRESSED_DATA_FILENAME, compressed_data)


def _scale_and_save_data(rundir, data_getter):
    """
    Create a file with compressed data.

    The compressed data contains compressed heterodyned data and
    processed_coef.
    It is a float32 array of shape (n_simulations, n_features).
    """
    training_data = _get_data(rundir/utils.TRAINING_DIR, data_getter)

    scaler = JSONStandardScaler()
    scaler.fit(training_data)
    scaler.to_json(rundir)

    _save_compressed_data(training_data, scaler, rundir/utils.TRAINING_DIR)
    del training_data

    test_data = _get_data(rundir/utils.TEST_DIR, data_getter)
    _save_compressed_data(test_data, scaler, rundir/utils.TRAINING_DIR)


def simple_compression(rundir):
    """
    No compression other than the heterodyning itself.

    Create files with compressed data for the training and test sets.
    The compressed data contains flattened heterodyned data (real and
    imaginary parts) and processed_coef.
    It is a float32 array of shape (n_simulations, n_features).
    """
    rundir = Path(rundir)
    utils.check_version(rundir)

    def data_getter(datadir):
        with np.load(datadir/utils.PREPROCESSED_DATA_FILENAME) as file:
            heterodyned_data = file['heterodyned_data']

        n_sim, n_det, n_freq = heterodyned_data.shape
        return heterodyned_data.reshape(n_sim, n_det * n_freq)

    _scale_and_save_data(rundir, data_getter)


def svd_compression(rundir, target_loss=1e-3):
    """
    Compress the heterodyned data using SVD.

    Create files with compressed data for the training and test sets.
    The compressed data contains SVD coefficients and processed_coef.
    It is a float32 array of shape (n_simulations, n_features).

    Parameters
    ----------
    rundir: os.PathLike
        Path to run directory.

    target_loss: float between 0 and 1
        How much information we afford to discard, in terms of the
        fractional variance of the Wiener-filtered signal. Smaller is
        more conservative, at the expense of less compression.
    """
    rundir = Path(rundir)
    utils.check_version(rundir)
    compressor = SVDCompressor.from_training_data(rundir)
    n_components = compressor.n_components(target_loss)

    def data_getter(datadir):
        data, _ = SVDCompressor.load_data_and_signal(datadir,
                                                     apply_mask=False)
        return compressor.get_svd_coefficients(data, n_components)

    _scale_and_save_data(rundir, data_getter)

    compressor.to_npz(rundir)


class SVDCompressor(utils.NpzMixin):
    """Class to compress data using SVD."""
    @classmethod
    def from_training_data(cls, rundir):
        """
        Load heterodyned data and signal, apply mask and construct SVD.

        Parameters
        ----------
        rundir: os.PathLike
            Directory with preprocessed data.
        """
        # We will never want to create a compressor using the test data
        datadir = Path(rundir)/utils.TRAINING_DIR

        # Load heterodyned data (noisy) and signal (noiseless):
        data, signal = cls.load_data_and_signal(datadir)
        noise = data - signal

        mean_signal = np.mean(signal, axis=0)
        std_noise = np.std(noise, axis=0)

        # "Whiten" the components using the measured spectrum
        wht_signal = (signal - mean_signal) / std_noise
        wht_noise = noise / std_noise

        # Construct SVD bases using the (whitened) signals:
        vh_mat = np.linalg.svd(wht_signal, full_matrices=False).Vh

        # Construct Wiener filter
        signal_svd_coef = wht_signal @ vh_mat.conjugate().transpose()
        noise_svd_coef = wht_noise @ vh_mat.conjugate().transpose()

        signal_spectrum = np.var(signal_svd_coef, axis=0)
        noise_spectrum = np.var(noise_svd_coef, axis=0)

        wiener_filter = signal_spectrum / (signal_spectrum + noise_spectrum)

        # Cumulative variance of the Wiener-filtered signal, useful to
        # later decide how many SVD components we should keep
        cumulative_variance = np.cumsum(wiener_filter**2 * signal_spectrum)
        cumulative_variance /= cumulative_variance[-1]

        return cls(mean_signal, std_noise, vh_mat, cumulative_variance)

    def __init__(self, _mean_signal, _std_noise, _vh_mat,
                 _cumulative_variance):
        """
        This is a generic constructor, use ``.from_training_data`` or
        ``.from_npz`` instead.
        """
        self._mean_signal = _mean_signal
        self._std_noise = _std_noise
        self._vh_mat = _vh_mat
        self._cumulative_variance = _cumulative_variance

    def get_svd_coefficients(self, data, n_components=None):
        """
        Return coefficients describing the data in the SVD basis.

        Parameters
        ----------
        data: array of shape (?, n_features)
            Heterodyned data to compress, e.g. from the output of
            ``.load_data_and_signal``.

        n_components: int
            How many components to keep, e.g. from the output of
            ``.n_components``. ``None`` (default) is no compression.

        Returns
        -------
        svd_coef: array of the same shape and type as `data`
            The first columns capture most of the variance of the
            whitened signal; compression is achieved by dropping the
            last columns.
        """
        wht_data = (data - self._mean_signal) / self._std_noise
        svd_coef = wht_data @ self._vh_mat.conjugate().transpose()
        return svd_coef[..., :n_components]

    def n_components(self, target_loss=1e-3) -> int:
        """
        Return number of SVD components needed in order to preserve the
        signal up to an acceptable loss.

        Parameters
        ----------
        target_loss: float between 0 and 1
            How much information we afford to discard, in terms of the
            fractional variance of the Wiener-filtered signal.
        """
        return np.searchsorted(self._cumulative_variance, 1-target_loss) + 1

    @staticmethod
    def load_data_and_signal(datadir, apply_mask=True):
        """Return heterodyned data and signal, reshaped for this class."""
        preprocessed_data = utils.get_preprocessed_data(datadir, apply_mask)

        n_sim, n_det, n_freq = preprocessed_data['heterodyned_data'].shape
        shape = n_sim, n_det*n_freq

        complex_data = preprocessed_data['heterodyned_data'].reshape(shape)
        complex_signal = preprocessed_data['heterodyned_signal'].reshape(shape)

        data = np.concatenate([complex_data.real, complex_data.imag], axis=1)
        signal = np.concatenate([complex_signal.real, complex_signal.imag],
                                axis=1)
        return data, signal


class JSONStandardScaler(sklearn.preprocessing.StandardScaler):
    """
    Like ``sklearn.preprocessing.StandardScaler`` but it can be saved to
    JSON.
    """
    _KEYS = ('mean_',
             'var_',
             'scale_',
             'n_samples_seen_')

    @classmethod
    def from_json(cls, directory):
        """Load the ``StandardScaler`` parameters from a JSON file."""
        filepath = cls._get_filepath(directory)

        with open(filepath, encoding='utf-8') as file:
            scaler_params = json.load(file)

        scaler = cls()
        scaler.__dict__.update(scaler_params)
        return scaler

    def to_json(self, directory):
        """Save the ``StandardScaler`` parameters to a JSON file."""
        filepath = self._get_filepath(directory)

        scaler_params = {key: getattr(self, key) for key in self._KEYS}

        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(scaler_params, file, cls=cogwheel.utils.NumpyEncoder)

    @classmethod
    def _get_filepath(cls, directory):
        return Path(directory) / f'{cls.__name__}.json'


def submit_condor(rundir,
                  compression_algorithm='svd_compression',
                  request_cpus=1,
                  request_memory='25G',
                  request_disk='1G',
                  **submit_kwargs):
    """
    Submit an HTCondor job to compress data.

    This will generate the following files:
        {rundir}/submission_scripts/compression.{sub,sh,out,err,log}

    Parameters
    ----------
    rundir: str, os.PathLike
        Simulations directory, on which `simulation` has already
        been run.

    request_cpus, request_memory, request_disk: int or str
        Specifications in the HTCondor submit file.

    **submit_kwargs
        Further options to include in the HTCondor submit file. Do
        not pass `executable`, `output`, `error`, `log`, `args`,
        `queue`, which will be dealt with automatically.
    """
    rundir = Path(rundir).resolve()
    scripts_dir = rundir/'submission_scripts'
    os.makedirs(scripts_dir, exist_ok=True)

    submit_kwargs = {
        'submit_path': scripts_dir/'compression.sub',
        'executable': scripts_dir/'compression.sh',
        'output': scripts_dir/'compression.out',
        'error': scripts_dir/'compression.err',
        'log': scripts_dir/'compression.log',
        'args': f'{rundir} {compression_algorithm}',
        'request_cpus': request_cpus,
        'request_memory': request_memory,
        'request_disk': request_disk,
        } | submit_kwargs

    cogwheel.utils.submit_condor(**submit_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress data.')

    parser.add_argument('rundir',
                        help='''Simulation directory path, on which
                                `simulation` has already been run.''')


    parser.add_argument('compression_algorithm', type=str,
                        help='"simple_compression" or "svd_compression".')

    args = parser.parse_args()
    # Get compression function by name
    compress = getattr(sys.modules[__name__], args.compression_algorithm)
    compress(args.rundir)
