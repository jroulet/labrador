"""
Algorithms for compressing the preprocessed data, and for defining a
mask to select a subset of the simulations.
"""
from pathlib import Path
import numpy as np
from cogwheel_machine.generate_parameters import load_config, CONFIG_FILENAME


COMPRESSED_DATA_FILENAME = 'compressed_data.npy'
MASK_FILENAME = 'mask.npy'


def simple_compression(sim_dir):
    """
    Create a file ``{sim_dir}/{COMPRESSED_DATA_FILENAME}`` with
    compressed data.

    The compressed data contains flattened heterodyned data (real &
    imaginary parts) and processed_coef.
    It is a float32 array of shape (n_simulations, n_features).
    """
    sim_dir = Path(sim_dir)

    preprocessed_data = np.load(sim_dir/'preprocessed_data.npz')

    heterodyned_data = preprocessed_data['heterodyned_data']
    n_sim, n_det, n_freq = heterodyned_data.shape
    reshaped = heterodyned_data.reshape(n_sim, n_det * n_freq)

    compressed_data = np.concatenate(
        [reshaped.real,
         reshaped.imag,
         preprocessed_data['processed_coef']
        ], dtype=np.float32, axis=1)
    np.save(sim_dir/COMPRESSED_DATA_FILENAME, compressed_data)


def create_snr_mask(sim_dir):
    """
    Save a mask in {sim_dir}/{MASK_FILENAME} specifying which
    simulations satisfy the SNR cuts.

    The mask is a boolean array of shape (n_simulations,) encoding
    which simulations satisfy the SNR cuts.

    Parameters
    ----------
    sim_dir: os.PathLike
        Path to simulation directory.

    snr0_min, snr0_max: float
        Minimum and maximum values for the semicoherent SNR of the
        reference waveform.
    """
    sim_dir = Path(sim_dir)

    config = load_config(sim_dir/CONFIG_FILENAME)
    preprocessed_data = np.load(sim_dir/'preprocessed_data.npz')

    d_h0_semicoherent = np.sum(preprocessed_data['d_h0_semicoherent'], axis=1)
    h0_h0 = np.sum(preprocessed_data['h0_h0'], axis=1)
    snr0 = d_h0_semicoherent / np.sqrt(h0_h0)

    snr0_min, snr0_max = config.SNR0_RANGE

    mask = (snr0 >= snr0_min) & (snr0 <= snr0_max)
    np.save(sim_dir/MASK_FILENAME, mask)
