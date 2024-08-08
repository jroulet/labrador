"""Utility functions and constants."""

import logging
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

import cogwheel.validation

from cogwheel_machine import __version__
from cogwheel_machine import config as example_config


CONFIG_FILENAME = 'config.py'
PARAMETERS_FILENAME = 'simulation_parameters.feather'
PREPROCESSED_DATA_FILENAME = 'preprocessed_data.npz'
FOLDED_SAMPLED_PARAMS_FILENAME = 'folded_sampled_params.npy'
UNFOLDING_LABELS_FILENAME = 'unfolding_labels.npy'
MASK_FILENAME = 'mask.npy'
COMPRESSED_DATA_FILENAME = 'compressed_data.npy'
VERSION_FILENAME = 'version.txt'


def load_config(sim_dir):
    """Return module `config` from a simulations directory."""
    return cogwheel.validation.load_config(sim_dir/CONFIG_FILENAME)


def setup_sim_dir(location, prefix='set_'):
    """
    Set up a simulations directory with an example config.py file.


    Parameters
    ----------
    location: os.PathLike
        Path in which to create the simulations directory ``sim_dir``.

    prefix: str
        ``sim_dir`` will be named as the prefix follwed by a number, to
        make it unique.

    Returns
    -------
    sim_dir: os.PathLike
        Path to the newly created simulations directory.
    """
    # Choose a unique name for the simulations directory
    location = Path(location)
    counter = 0
    while (sim_dir := location/f'{prefix}{counter}').exists():
        counter += 1

    os.makedirs(sim_dir)
    source = Path(example_config.__file__)
    destination = (sim_dir/CONFIG_FILENAME).resolve()
    shutil.copyfile(source, destination)

    print(f'Created a new config file at {destination}. Edit it as needed.')

    return sim_dir


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
    with np.load(sim_dir/PREPROCESSED_DATA_FILENAME) as preprocessed_data:
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


def get_preprocessed_data(sim_dir, apply_mask=True):
    """
    Load ``preprocessed_data`` and apply the ``mask`` to it.

    Parameters
    ----------
    sim_dir: os.PathLike
        Directory with training data.

    apply_mask: bool
        Whether to apply the mask in {sim_dir}/{MASK_FILENAME} to the
        loaded arrays.

    Returns
    -------
    dict: keys match those of ``preprocessed_data``.
    """
    sim_dir = Path(sim_dir)

    mask = None
    if apply_mask:
        mask = np.load(sim_dir/MASK_FILENAME)

    preprocessed_data = {}
    with np.load(sim_dir/PREPROCESSED_DATA_FILENAME) as file:
        for key, arr in file.items():
            if key == 'fbin' or not apply_mask:
                preprocessed_data[key] = arr
            else:
                preprocessed_data[key] = arr[mask]

    return preprocessed_data


def check_version(sim_dir):
    """
    Check that the version of cogwheel_machine recorded in `sim_dir`
    matches the current one.

    Issue a warning if not. Raise ``FileNotFoundError`` if `sim_dir`
    does not contain a version file.
    """
    sim_dir = Path(sim_dir)
    with open(sim_dir/VERSION_FILENAME, encoding='utf-8') as file:
        version = file.read()

    if version != __version__:
        logging.warning(f'{sim_dir} was populated using a different version of'
                        f' `cogwheel_machine`, {version!r}. '
                        f'The current version is {__version__!r}.')


def write_version(sim_dir):
    """Write the version of cogwheel_machine to a file in `sim_dir`."""
    sim_dir = Path(sim_dir)
    with open(sim_dir/VERSION_FILENAME, 'w', encoding='utf-8') as file:
        file.write(__version__)
