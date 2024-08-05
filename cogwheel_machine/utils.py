"""Utility functions and constants."""

from pathlib import Path
import numpy as np
import pandas as pd

import cogwheel.validation


CONFIG_FILENAME = 'config.py'
PARAMETERS_FILENAME = 'simulation_parameters.feather'
PREPROCESSED_DATA_FILENAME = 'preprocessed_data.npz'
FOLDED_SAMPLED_PARAMS_FILENAME = 'folded_sampled_params.npy'
UNFOLDING_LABELS_FILENAME = 'unfolding_labels.npy'
MASK_FILENAME = 'mask.npy'
COMPRESSED_DATA_FILENAME = 'compressed_data.npy'


def load_config(sim_dir):
    """Return module `config` from a simulations directory."""
    return cogwheel.validation.load_config(sim_dir/CONFIG_FILENAME)


def get_summary(sim_dir, apply_mask=True):
    """
    Return DataFrame with injection parameters, SNR, and parameters in
    the target space of the normalizing flow.
    """
    sim_dir = Path(sim_dir)
    config = load_config(sim_dir)

    # Injection parameters
    summary = pd.read_feather(sim_dir/PARAMETERS_FILENAME)

    # Add SNR
    preprocessed_data = np.load(sim_dir/PREPROCESSED_DATA_FILENAME)
    for key in 'd_h', 'h_h', 'd_h0_semicoherent', 'h0_h0':
        summary[key] = preprocessed_data[key].sum(axis=1)
    summary['snr'] = summary['d_h'] / np.sqrt(summary['h_h'])
    summary['snr0'] = summary['d_h0_semicoherent'] / np.sqrt(summary['h0_h0'])

    # Add transformed parameters
    columns = list(config.TRANSFORM_CLASS.sampled_params)
    for par in config.TRANSFORM_CLASS.folded_params:
        columns[columns.index(par)] = f'folded_{par}'
    folded_sampled_params = pd.DataFrame(
        np.load(sim_dir/FOLDED_SAMPLED_PARAMS_FILENAME), columns=columns)
    cogwheel.utils.update_dataframe(summary, folded_sampled_params)

    # Apply mask
    if apply_mask:
        mask = np.load(sim_dir/MASK_FILENAME)
        summary = summary[mask]

    return summary


def get_masked_preprocessed_data(sim_dir):
    """
    Load preprocessed_data and apply the mask to it.

    Parameters
    ----------
    sim_dir: os.PathLike
        Directory with training data.

    Returns
    -------
    dict: keys match those of the `compressed_data` structured array.
    """
    sim_dir = Path(sim_dir)

    mask = np.load(sim_dir/MASK_FILENAME)

    preprocessed_data = {}
    for key, arr in np.load(sim_dir/PREPROCESSED_DATA_FILENAME).items():
        if key == 'fbin':
            preprocessed_data[key] = arr
        else:
            preprocessed_data[key] = arr[mask]

    return preprocessed_data
