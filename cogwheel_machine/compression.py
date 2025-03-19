"""
Algorithms for compressing the preprocessed data, and for defining a
mask to select a subset of the simulations.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import h5py
import torch
import sklearn.preprocessing
from sklearn.utils.extmath import randomized_svd

import cogwheel.utils

from . import utils


def create_mask(rundir):
    """
    Create and save masks for the training and test data, specifying
    which simulations satisfy ``data_config.MASK_CONDITIONS``.

    The mask is a boolean array of shape (n_simulations,) encoding which
    simulations satisfy the cuts.

    Parameters
    ----------
    rundir : os.PathLike
        Path to run directory.
    """
    rundir = Path(rundir)
    utils.check_version(rundir)

    data_config = utils.load_data_config(rundir)

    for datadir in rundir/utils.TRAINING_DIR, rundir/utils.TEST_DIR:
        summary = utils.get_summary(datadir, apply_mask=False)

        mask = np.full(len(summary), True)
        for par, logic, value in data_config.MASK_CONDITIONS:
            mask &= logic(summary[par], value)

        np.save(datadir/utils.MASK_FILENAME, mask)


def svd_compression(rundir, target_loss=1e-3, max_svd_size=100_000,
                    chunk_size=10_000):
    """
    Compress the heterodyned data using SVD.

    Create files with compressed data for the training and test sets.
    The compressed data contains SVD coefficients and processed_coef.
    It is a float32 array of shape (n_simulations, n_features).

    Parameters
    ----------
    rundir : os.PathLike
        Path to run directory.

    target_loss : float between 0 and 1
        How much information we afford to discard, in terms of the
        fractional variance of the Wiener-filtered signal. Smaller is
        more conservative, at the expense of less compression.

    max_svd_size : int
        Maximum number of preprocessed data examples to input to the
        SVD. Mostly for memory considerations.

    chunk_size : int
        The preprocessed data will be compressed in chunks, to avoid
        loading it all to memory.
    """
    rundir = Path(rundir)
    utils.check_version(rundir)

    # Define the SVD basis
    compressor = SVDCompressor.from_training_data(rundir, max_svd_size,
                                                  target_loss)
    compressor.to_npz(rundir)

    training_data = _load_unscaled_data(rundir/utils.TRAINING_DIR, chunk_size)
    mask = np.load(rundir/utils.TRAINING_DIR/utils.MASK_FILENAME)

    # Fit scaler
    scaler = JSONStandardScaler()
    scaler.fit(training_data[mask])
    scaler.to_json(rundir)

    _save_compressed_data(training_data, scaler, rundir/utils.TRAINING_DIR)
    del training_data, mask

    test_data = _load_unscaled_data(rundir/utils.TEST_DIR, chunk_size)
    _save_compressed_data(test_data, scaler, rundir/utils.TEST_DIR)


def _load_unscaled_data(datadir, chunk_size=10_000):
    """
    This is `compressed_data`, but before passing through `data_scaler`.
    """
    compressor = SVDCompressor.from_npz(datadir.parent)

    with h5py.File(datadir/utils.PREPROCESSED_DATA_FILENAME, "r") as file:
        heterodyned_data = file['heterodyned_data']

        # Load heterodyned data in chunks to save memory
        size = len(heterodyned_data)
        slices = [slice(chunk_start, chunk_start + chunk_size)
                  for chunk_start in range(0, size, chunk_size)]
        svd_chunks = []
        for slice_ in slices:
            svd_chunks.append(compressor.reshape_and_project_data(
                heterodyned_data[slice_]))

        # processed_coef should be lightweight
        processed_coef = file['processed_coef'][:]

    svd_coef = np.concatenate(svd_chunks, axis=0)
    del svd_chunks

    return np.concatenate([svd_coef, processed_coef], axis=1)


def _save_compressed_data(unscaled_data, scaler, datadir):
    compressed_data = scaler.transform(unscaled_data).astype(np.float32)
    np.save(datadir/utils.COMPRESSED_DATA_FILENAME, compressed_data)


def compress_data(rundir, heterodyned_data, processed_coef):
    """Convenience function for interfacing with other modules."""
    # Perhaps this function should go somewhere more accessible.
    compressor = SVDCompressor.from_npz(rundir)
    data_scaler = JSONStandardScaler.from_json(rundir)
    svd_coef = compressor.reshape_and_project_data(heterodyned_data)
    unscaled_data = np.concatenate([svd_coef, processed_coef], axis=-1)
    compressed_data = data_scaler.transform(np.atleast_2d(unscaled_data))

    return torch.from_numpy(compressed_data).to(torch.float32)


class SVDCompressor(utils.NpzMixin):
    """Class to compress data using SVD."""
    @classmethod
    def from_training_data(cls, rundir, max_svd_size=None,
                           target_loss=1e-3):
        """
        Load heterodyned data and signal, apply mask and construct SVD.

        Parameters
        ----------
        rundir : os.PathLike
            Directory with preprocessed data.

        max_svd_size : int
            Maximum number of preprocessed data examples to input to the
            SVD. Mostly for memory considerations.

        target_loss : float between 0 and 1
            How much information we afford to discard, in terms of the
            fractional variance of the Wiener-filtered signal.
        """
        # We will never want to create a compressor using the test data
        datadir = Path(rundir)/utils.TRAINING_DIR

        # Load heterodyned data (noisy) and signal (noiseless):
        data, signal = cls.load_data_and_signal(
            datadir, slice_=slice(max_svd_size))
        noise = data - signal

        mean_signal = np.mean(signal, axis=0)
        std_noise = np.std(noise, axis=0)

        # "Whiten" the components using the measured spectrum
        wht_signal = (signal - mean_signal) / std_noise
        wht_noise = noise / std_noise

        # Construct SVD bases using the (whitened) signals:
        _, _, vh_mat = randomized_svd(wht_signal, n_components=100)

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

        return cls(mean_signal, std_noise, vh_mat, cumulative_variance,
                   target_loss)

    def __init__(self, _mean_signal, _std_noise, _vh_mat,
                 _cumulative_variance, _target_loss):
        """
        This is a generic constructor, use ``.from_training_data`` or
        ``.from_npz`` instead.
        """
        self._mean_signal = _mean_signal
        self._std_noise = _std_noise
        self._vh_mat = _vh_mat
        self._cumulative_variance = _cumulative_variance
        self._target_loss = _target_loss

    def get_svd_coefficients(self, data, n_components=None):
        """
        Return coefficients describing the data in the SVD basis.

        Parameters
        ----------
        data : array of shape (?, n_features)
            Heterodyned data to compress, e.g. from the output of
            ``.load_data_and_signal``.

        n_components : int
            How many components to keep, e.g. from the output of
            ``.n_components``. ``None`` (default) is the number of
            components ``._ncomponents`` (defined by ``._target_loss``).

        Returns
        -------
        svd_coef : array of the same shape and type as `data`
            The first columns capture most of the variance of the
            whitened signal; compression is achieved by dropping the
            last columns.
        """
        if n_components is None:
            n_components = self.n_components

        wht_data = (data - self._mean_signal) / self._std_noise
        return wht_data @ self._vh_mat[:n_components].conjugate().transpose()

    @property
    def n_components(self) -> int:
        """
        Number of SVD components needed in order to preserve the signal
        up to an acceptable loss.
        """
        return 1 + np.searchsorted(self._cumulative_variance,
                                   1.0 - self._target_loss)

    @classmethod
    def load_data_and_signal(cls, datadir, apply_mask=True,
                             slice_=slice(None)):
        """
        Return heterodyned data and signal, reshaped for this class.
        """
        preprocessed_data = utils.get_preprocessed_data(datadir, apply_mask,
                                                        slice_)

        data = cls.reshape_heterodyned_data(
            preprocessed_data.pop('heterodyned_data'))

        signal = cls.reshape_heterodyned_data(
            preprocessed_data.pop('heterodyned_signal'))

        return data, signal

    @staticmethod
    def reshape_heterodyned_data(heterodyned_data):
        """
        Reshape heterodyned data or signal for this class.

        Parameters
        ----------
        heterodyned_data : (n_sim?, n_det, n_freq) complex array
            Could be one of ``preprocessed_data['heterodyned_data']`` or
            ``preprocessed_data['heterodyned_signal']``.

        Returns
        -------
        (n_sim?, 2 * n_det * n_freq) float array
        """
        *n_sim, n_det, n_freq = heterodyned_data.shape
        shape = *n_sim, n_det * n_freq

        complex_data = heterodyned_data.reshape(shape)
        return np.concatenate([complex_data.real, complex_data.imag], axis=-1)

    def reshape_and_project_data(self, heterodyned_data,
                                 n_components=None):
        """Return SVD coefficients of the heterodyned data."""
        reshaped_data = self.reshape_heterodyned_data(heterodyned_data)
        return self.get_svd_coefficients(reshaped_data, n_components)


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
            json.dump(scaler_params, file, cls=cogwheel.utils.NumpyEncoder,
                      indent=2)

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
    rundir : str, os.PathLike
        Simulations directory, on which `simulation` has already
        been run.

    request_cpus, request_memory, request_disk : int or str
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

    args = parser.parse_args()
    svd_compression(args.rundir)
