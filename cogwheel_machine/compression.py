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


def simple_compression(sim_dir):
    """
    Create a file ``{sim_dir}/{COMPRESSED_DATA_FILENAME}`` with
    compressed data.

    The compressed data contains flattened heterodyned data (real &
    imaginary parts) and processed_coef.
    It is a float32 array of shape (n_simulations, n_features).
    """
    sim_dir = Path(sim_dir)
    utils.check_version(sim_dir)

    preprocessed_data = np.load(sim_dir/utils.PREPROCESSED_DATA_FILENAME)

    heterodyned_data = preprocessed_data['heterodyned_data']
    n_sim, n_det, n_freq = heterodyned_data.shape
    reshaped = heterodyned_data.reshape(n_sim, n_det * n_freq)

    compressed_data = np.concatenate(
        [reshaped.real,
         reshaped.imag,
         preprocessed_data['processed_coef']
        ], dtype=np.float32, axis=1)
    np.save(sim_dir/utils.COMPRESSED_DATA_FILENAME, compressed_data)
