"""
Utility functions and constants.

File structure:
The final file structure of a trained model should look as below. The
user only edits the files `data_config.py` and `model_config.py` by
hand, all the rest are created by the various modules of the code.

{parentdir}/                             # E.g. 'coghweel-machine/data/'
└── {rundir}/                            # E.g. 'run_0'
    ├── {datadir}/                       # 'training_data' or 'test_data'
    │   ├── compressed_data.npy
    │   ├── folded_sampled_parameters.npy
    │   ├── mask.npy
    │   ├── preprocessed_data.npz
    │   ├── simulation_parameters.feather
    │   └── unfolding_labels.npy
    ├── {modeldir}/                      # E.g. 'model_0'
    │   ├── model_config.py
    │   └── posterior.pt
    ├── data_config.py
    └── version.txt
"""

import functools
import logging
import multiprocessing
import os
import pstats
import shutil
import sys
import tempfile
from pathlib import Path
from cProfile import Profile
import numpy as np
import pandas as pd

import cogwheel.utils
import cogwheel.validation

from cogwheel_machine import __version__


EXAMPLE_CONFIGS_DIR = Path(__file__).parent/'example_configs'
TRAINING_DIR = 'training_data'
TEST_DIR = 'test_data'
DATA_CONFIG_FILENAME = 'data_config.py'
MODEL_CONFIG_FILENAME = 'model_config.py'
PARAMETERS_FILENAME = 'simulation_parameters.feather'
PREPROCESSED_DATA_FILENAME = 'preprocessed_data.npz'
FOLDED_SAMPLED_PARAMS_FILENAME = 'folded_sampled_params.npy'
UNFOLDING_LABELS_FILENAME = 'unfolding_labels.npy'
MASK_FILENAME = 'mask.npy'
COMPRESSED_DATA_FILENAME = 'compressed_data.npy'
VERSION_FILENAME = 'version.txt'
RESCALED_PARAMETERS_FILENAME = 'rescaled_parameters.npy'
INFERENCE_FILENAME = 'inference.pickle'
POSTERIOR_FILENAME = 'posterior.pt'


def load_data_config(rundir):
    """Return module `data_config` from a run directory."""
    rundir = Path(rundir)
    with cogwheel.utils.temporarily_change_attributes(
            sys, dont_write_bytecode=True):  # TODO move to cogwheel
        return cogwheel.validation.load_config(rundir/DATA_CONFIG_FILENAME)


def load_model_config(modeldir):
    """Return module `model_config` from a model directory."""
    modeldir = Path(modeldir)
    with cogwheel.utils.temporarily_change_attributes(
            sys, dont_write_bytecode=True):  # TODO move to cogwheel
        return cogwheel.validation.load_config(modeldir/MODEL_CONFIG_FILENAME)


def make_unique_dir(location, prefix):
    """
    Make a new directory inside `location` ensuring it has a unique
    name of the form `{prefix}{counter}`.

    Return a ``pathlib.Path`` object pointing to that directory.
    """
    location = Path(location)
    counter = 0
    while (dirname := location/f'{prefix}{counter}').exists():
        counter += 1
    os.makedirs(dirname)
    return dirname


def setup_rundir(parentdir, prefix='run_'):
    """
    Set up a run directory with an example data_config.py file.

    Parameters
    ----------
    parentdir: os.PathLike
        Path in which to create the run directory ``rundir``.

    prefix: str
        ``rundir`` will be named as the prefix follwed by a number, to
        make it unique.

    Returns
    -------
    rundir: os.PathLike
        Path to the newly created run directory.
    """
    rundir = make_unique_dir(parentdir, prefix)

    source = EXAMPLE_CONFIGS_DIR/DATA_CONFIG_FILENAME
    destination = (rundir/DATA_CONFIG_FILENAME).resolve()
    shutil.copyfile(source, destination)

    print(f'Created a new data config file at {destination}.',
          'Edit it as needed.')

    return rundir


def setup_modeldir(rundir, prefix='model_'):
    """
    Set up a model directory with an example model_config.py file.

    Parameters
    ----------
    rundir: os.PathLike
        Path in which to create the model directory ``modeldir``.

    prefix: str
        ``modeldir`` will be named as the prefix followed by a number, to
        make it unique.

    Returns
    -------
    modeldir: os.PathLike
        Path to the newly created model directory.
    """
    modeldir = make_unique_dir(rundir, prefix)

    source = EXAMPLE_CONFIGS_DIR/MODEL_CONFIG_FILENAME
    destination = (modeldir/MODEL_CONFIG_FILENAME).resolve()
    shutil.copyfile(source, destination)

    print(f'Created a new model config file at {destination}.',
          'Edit it as needed.')
    return modeldir


def get_summary(datadir, apply_mask=True):
    """
    Return DataFrame with injection parameters, SNR, and parameters in
    the target space of the normalizing flow.

    Parameters
    ----------
    datadir: os.PathLike
        Path to the run directory in which training or test data have
        been created.

    apply_mask: bool
        Whether to apply the boolean mask to the data.
    """
    datadir = Path(datadir)
    config = load_data_config(datadir.parent)

    # Injection parameters
    summary = pd.read_feather(datadir/PARAMETERS_FILENAME)

    # Add SNR
    with np.load(datadir/PREPROCESSED_DATA_FILENAME) as preprocessed_data:
        for key in 'd_h', 'h_h', 'd_h0_semicoherent', 'h0_h0':
            summary[key] = preprocessed_data[key].sum(axis=1)

    summary['snr'] = summary['d_h'] / np.sqrt(summary['h_h'])
    summary['snr0'] = summary['d_h0_semicoherent'] / np.sqrt(summary['h0_h0'])

    # Add transformed parameters
    columns = list(config.TRANSFORM_CLASS.sampled_params)
    for par in config.TRANSFORM_CLASS.folded_params:
        columns[columns.index(par)] = f'folded_{par}'
    folded_sampled_params = pd.DataFrame(
        np.load(datadir/FOLDED_SAMPLED_PARAMS_FILENAME), columns=columns)
    cogwheel.utils.update_dataframe(summary, folded_sampled_params)

    # Apply mask
    if apply_mask:
        mask = np.load(datadir/MASK_FILENAME)
        summary = summary[mask]

    return summary


def get_preprocessed_data(datadir, apply_mask=True) -> dict:
    """
    Load ``preprocessed_data`` and apply the ``mask`` to it.

    Parameters
    ----------
    datadir: os.PathLike
        Path to the run directory in which training or test data have
        been created.

    apply_mask: bool
        Whether to apply the mask in {datadir}/{MASK_FILENAME} to the
        loaded arrays.

    Returns
    -------
    dict: keys match those of ``preprocessed_data``.
    """
    mask = None
    if apply_mask:
        mask = np.load(datadir/MASK_FILENAME)

    preprocessed_data = {}
    with np.load(datadir/PREPROCESSED_DATA_FILENAME) as file:
        for key, arr in file.items():
            if key == 'fbin' or not apply_mask:
                preprocessed_data[key] = arr
            else:
                preprocessed_data[key] = arr[mask]

    return preprocessed_data


def check_version(rundir):
    """
    Check that the version of cogwheel_machine recorded in `rundir`
    matches the current one.

    Issue a warning if not. Raise ``FileNotFoundError`` if `rundir`
    does not contain a version file.
    """
    rundir = Path(rundir)
    with open(rundir/VERSION_FILENAME, encoding='utf-8') as file:
        version = file.read()

    if version != __version__:
        logging.warning(f'{rundir} was populated using a different version of'
                        f' `cogwheel_machine`, {version!r}. '
                        f'The current version is {__version__!r}.')


def write_version(rundir):
    """Write the version of cogwheel_machine to a file in `rundir`."""
    rundir = Path(rundir)
    with open(rundir/VERSION_FILENAME, 'w', encoding='utf-8') as file:
        file.write(__version__)


class NpzMixin:
    """
    Implement ``.from_npz``, ``.to_npz`` and ``get_filename`` for
    classes that only contain numpy.array attributes.
    """
    @classmethod
    def from_npz(cls, directory):
        """Load instance from a .npz file."""
        with np.load(cls.get_filename(directory)) as file:
            return cls(**file)

    def to_npz(self, directory):
        """Save instance to a .npz file."""
        np.savez(self.get_filename(directory), **self.__dict__)

    @classmethod
    def get_filename(cls, directory):
        """
        Return path to a .npz file in directory, defining a convention
        for where to save instances of this class.
        """
        return Path(directory)/f'{cls.__name__}.npz'


def multiprocessing_starmap_profiled(func, iterable, processes=None):
    """
    Similar to ``multiprocessing.Pool().starmap`` but it also returns
    profiling statistics.

    Return
    ------
    results: list
        ``[func(*args) for args in iterable]``.

    stats: pstats.Stats
        Profiling statistics.
    """
    with tempfile.TemporaryDirectory() as profile_dir:
        profiled_func = functools.partial(_aux_profiled_func,
                                          func=func, profile_dir=profile_dir)

        with multiprocessing.Pool(processes, _worker_initializer) as pool:
            results = pool.map(profiled_func, iterable)

        # Aggregate the stats
        paths = (path.as_posix() for path in Path(profile_dir).glob('*.prof'))
        stats = pstats.Stats(*paths)

    return results, stats

def _worker_initializer():
    global profiler
    profiler = Profile()

def _aux_profiled_func(args, func, profile_dir):
    # Defined in top level so that it is pickleable for multiprocessing
    # global profiler
    result = profiler.runcall(func, *args)

    # Dump profile data after each call
    process_id = multiprocessing.current_process().pid
    profiler.dump_stats(Path(profile_dir)/f'{process_id}.prof')

    return result
