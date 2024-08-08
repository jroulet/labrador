"""
Algorithms for compressing the preprocessed data, and for defining a
mask to select a subset of the simulations.
"""
from pathlib import Path
import numpy as np

from . import utils


def create_mask(sim_dir):
    """
    Save a mask in {sim_dir}/{MASK_FILENAME} specifying which
    simulations satisfy the cuts per ``config.MASK_CONDITIONS``.

    The mask is a boolean array of shape (n_simulations,) encoding
    which simulations satisfy the cuts.

    Parameters
    ----------
    sim_dir: os.PathLike
        Path to simulation directory.
    """
    sim_dir = Path(sim_dir)
    utils.check_version(sim_dir)

    config = utils.load_config(sim_dir)
    summary = utils.get_summary(sim_dir, apply_mask=False)

    mask = np.ones(len(summary), dtype=bool)
    for par, logic, value in config.MASK_CONDITIONS:
        mask &= logic(summary[par], value)

    np.save(sim_dir/utils.MASK_FILENAME, mask)


def _save_compressed_data(sim_dir, compressed_heterodyned_data):
    """
    Create a file ``{sim_dir}/{COMPRESSED_DATA_FILENAME}`` with
    compressed data.
    The compressed data contains compressed heterodyned data and
    processed_coef.
    It is a float32 array of shape (n_simulations, n_features).
    """
    sim_dir = Path(sim_dir)
    utils.check_version(sim_dir)

    with np.load(sim_dir/utils.PREPROCESSED_DATA_FILENAME) as file:
        processed_coef = file['processed_coef']

    compressed_data = np.concatenate([compressed_heterodyned_data,
                                      processed_coef],
                                     axis=1, dtype=np.float32)

    np.save(sim_dir/utils.COMPRESSED_DATA_FILENAME, compressed_data)


def simple_compression(sim_dir):
    """
    No compression other than the heterodyning itself.

    Create a file ``{sim_dir}/{COMPRESSED_DATA_FILENAME}`` with
    compressed data.
    The compressed data contains flattened heterodyned data (real &
    imaginary parts) and processed_coef.
    It is a float32 array of shape (n_simulations, n_features).
    """
    sim_dir = Path(sim_dir)

    with np.load(sim_dir/utils.PREPROCESSED_DATA_FILENAME) as file:
        heterodyned_data = file['heterodyned_data']

    n_sim, n_det, n_freq = heterodyned_data.shape
    reshaped_heterodyned_data = heterodyned_data.reshape(n_sim, n_det * n_freq)

    _save_compressed_data(sim_dir, reshaped_heterodyned_data)


def svd_compression(sim_dir, target_loss=1e-3):
    """
    Compress the heterodyned data using SVD.

    Create a file ``{sim_dir}/{COMPRESSED_DATA_FILENAME}`` with
    compressed data.
    The compressed data contains SVD coefficients and processed_coef.
    It is a float32 array of shape (n_simulations, n_features).

    Parameters
    ----------
    target_loss: float between 0 and 1
        How much information we afford to discard, in terms of the
        fractional variance of the Wiener-filtered signal.
    """
    compressor = SVDCompressor(sim_dir)
    data, _ = SVDCompressor.load_data_and_signal(sim_dir,
                                                 apply_mask=False)
    n_components = compressor.n_components(target_loss)
    svd_coefficients = compressor.get_svd_coefficients(data, n_components)
    _save_compressed_data(sim_dir, svd_coefficients)


class SVDCompressor:
    """Class to compress data using SVD."""
    def __init__(self, sim_dir):
        """
        Load heterodyned data and signal, apply mask and construct SVD.

        Parameters
        ----------
        sim_dir: os.PathLike
            Directory with preprocessed data.
        """
        data, signal  = self.load_data_and_signal(sim_dir)
        noise = data - signal
        self._mean_signal = np.mean(signal, axis=0)
        self._std_noise = np.std(noise, axis=0)
        self._vh_mat = self._compute_vh_mat(signal)
        self._cumulative_variance = self._get_cumulative_variance(signal,
                                                                  noise)

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
    def load_data_and_signal(sim_dir, apply_mask=True):
        """Return heterodyned data and signal, reshaped for this class."""
        preprocessed_data = utils.get_preprocessed_data(sim_dir, apply_mask)

        n_sim, n_det, n_freq = preprocessed_data['heterodyned_data'].shape
        shape = n_sim, n_det*n_freq

        complex_data = preprocessed_data['heterodyned_data'].reshape(shape)
        complex_signal = preprocessed_data['heterodyned_signal'].reshape(shape)

        data = np.concatenate([complex_data.real, complex_data.imag], axis=1)
        signal = np.concatenate([complex_signal.real, complex_signal.imag],
                                axis=1)
        return data, signal

    def _compute_vh_mat(self, signal):
        wht_signal = (signal - self._mean_signal) / self._std_noise
        return np.linalg.svd(wht_signal, full_matrices=False).Vh

    def _get_cumulative_variance(self, signal, noise):
        """Relative cumulative variance of the Wiener-filtered signal."""
        signal_spectrum = self._get_svd_coef_spectrum(signal)
        noise_spectrum = self._get_svd_coef_spectrum(noise)

        wiener_filter = signal_spectrum / (signal_spectrum + noise_spectrum)

        cum_var_wf_signal = np.cumsum(wiener_filter**2 * signal_spectrum)
        return cum_var_wf_signal / cum_var_wf_signal[-1]

    def _get_svd_coef_spectrum(self, data):
        svd_coef = self.get_svd_coefficients(data)
        return np.var(svd_coef, axis=0)
