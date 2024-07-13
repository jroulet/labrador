"""Algorithms for compressing the preprocessed data."""
from pathlib import Path
import numpy as np


COMPRESSED_DATA_FILENAME = 'compressed_data.npy'


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
