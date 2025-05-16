"""
Utility functions and constants.

File structure:
The final file structure of a trained model should look as below.
The user only edits the files `data_config.py`, `rescaler_config.py`,
`sbi_config.py` and `unfolding_config.py` by hand, all the rest are
created by the various modules of the code.

{parentdir}/                                   # E.g. 'coghweel-machine/data/'
└── {rundir}/                                  # E.g. 'run_0'
    ├── data_config.py
    ├── JSONStandardScaler.json
    ├── SVDCompressor.npz
    ├── version.txt
    ├── {datadir}/                             # 'training_data' or 'test_data'
    │   ├── compressed_data.npy
    │   ├── folded_sampled_params.h5
    │   ├── mask.npy
    │   ├── preprocessed_data.h5
    │   ├── simulation_parameters.feather
    │   ├── simulation_profiling
    │   └── unfolding_labels.h5
    └── {priordir}/                            # Name of physical-prior class
        ├── {datadir}/                         # 'training_data' or 'test_data'
        │   ├── ln_prior_ratios.npy
        │   └── weights.npy
        └── {rescalerdir}/                     # E.g. 'rescaler_0'
            ├── parameter_rescaler.pth
            ├── parameter_rescaler_training.pth
            ├── rescaler_config.py
            ├── {datadir}/                     # 'training_data' or 'test_data'
            │   └── rescaled_params.npy
            ├── {sbidir}/                      # E.g. 'sbi_0'
            │   ├── posterior.pt
            │   └── sbi_config.py
            └── {unfolderdir}/                 # E.g. 'unfolder_0'
                ├── unfolding_classifier.ubj
                └── unfolding_config.py

"""

import functools
import logging
import multiprocessing
import os
import pstats
import shutil
import subprocess
import tempfile
from pathlib import Path
from cProfile import Profile
import torch
import numpy as np
import pandas as pd
import h5py

import cogwheel.utils
import cogwheel.validation

from cogwheel_machine import __version__


EXAMPLE_CONFIGS_DIR = Path(__file__).parent/'example_configs'
TRAINING_DIR = 'training_data'
TEST_DIR = 'test_data'
DATA_CONFIG_FILENAME = 'data_config.py'
RESCALER_CONFIG_FILENAME = 'rescaler_config.py'
SBI_CONFIG_FILENAME = 'sbi_config.py'
UNFOLDER_CONFIG_FILENAME = 'unfolder_config.py'
PARAMETERS_FILENAME = 'simulation_parameters.feather'
PREPROCESSED_DATA_FILENAME = 'preprocessed_data.h5'
FOLDED_SAMPLED_PARAMETERS_FILENAME = 'folded_sampled_parameters.h5'
UNFOLDING_LABELS_FILENAME = 'unfolding_labels.h5'
MASK_FILENAME = 'mask.npy'
COMPRESSED_DATA_FILENAME = 'compressed_data.npy'
VERSION_FILENAME = 'version.txt'
RESCALED_PARAMETERS_FILENAME = 'rescaled_parameters.npy'
INFERENCE_FILENAME = 'inference.pickle'
POSTERIOR_FILENAME = 'posterior.pt'
UNFOLDER_FILENAME = 'unfolding_classifier.ubj'
WAVEFORM_MODEL_FILENAME = 'waveform_model.h5'
WEIGHTS_FILENAME = 'weights.npy'


def load_data_config(rundir):
    """Return module `data_config` from a run directory."""
    rundir = Path(rundir)
    return cogwheel.validation.load_config(rundir/DATA_CONFIG_FILENAME)


def load_rescaler_config(rescalerdir):
    """Return module `rescaler_config` from a rescaler directory."""
    rescalerdir = Path(rescalerdir)
    return cogwheel.validation.load_config(
        rescalerdir/RESCALER_CONFIG_FILENAME)


def load_sbi_config(sbidir):
    """Return module `sbi_config` from a sbi directory."""
    sbidir = Path(sbidir)
    return cogwheel.validation.load_config(sbidir/SBI_CONFIG_FILENAME)


def load_unfolder_config(unfolderdir):
    """Return module `unfolder_config` from an unfolder directory."""
    unfolderdir = Path(unfolderdir)
    return cogwheel.validation.load_config(
        unfolderdir/UNFOLDER_CONFIG_FILENAME)


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
    parentdir : os.PathLike
        Path in which to create the run directory ``rundir``.

    prefix : str
        ``rundir`` will be named as the prefix follwed by a number, to
        make it unique.

    Returns
    -------
    rundir : os.PathLike
        Path to the newly created run directory.
    """
    rundir = make_unique_dir(parentdir, prefix)

    source = EXAMPLE_CONFIGS_DIR/DATA_CONFIG_FILENAME
    destination = (rundir/DATA_CONFIG_FILENAME).resolve()
    shutil.copyfile(source, destination)

    print(f'Created a new data config file at {destination}.',
          'Edit it as needed.')

    return rundir


def setup_rescalerdir(priordir, prefix='rescaler_'):
    """
    Set up a rescaler directory with an example rescaler_config.py file.

    Parameters
    ----------
    priordir : os.PathLike
        Path in which to create the rescaler directory ``rescalerdir``.

    prefix : str
        ``rescaler`` will be named as the prefix followed by a number,
        to make it unique.

    Returns
    -------
    rescalerdir : os.PathLike
        Path to the newly created rescaler directory.
    """
    rescalerdir = make_unique_dir(priordir, prefix)

    source = EXAMPLE_CONFIGS_DIR/RESCALER_CONFIG_FILENAME
    destination = (rescalerdir/RESCALER_CONFIG_FILENAME).resolve()
    shutil.copyfile(source, destination)

    print(f'Created a new rescaler config file at {destination}.',
          'Edit it as needed.')
    return rescalerdir


def get_priordirs(rundir):
    """
    Get directories for physical priors inside a `rundir`.

    Does not create the directories or check whether they exist.

    Parameters
    ----------
    rundir : os.PathLike
        Path in which to create the prior directories.

    Returns
    -------
    priordirs : list of pathlib.Path
    """
    rundir = Path(rundir)
    data_config = load_data_config(rundir)

    return [rundir/prior_cls.__name__
            for prior_cls in data_config.PHYSICAL_PRIOR_CLASSES]


def setup_sbidir(rescalerdir, prefix='sbi_'):
    """
    Set up a sbi directory with an example sbi_config.py file.

    Parameters
    ----------
    rescalerdir : os.PathLike
        Path in which to create the sbi directory ``sbidir``.

    prefix : str
        ``sbi`` will be named as the prefix followed by a number, to
        make it unique.

    Returns
    -------
    sbidir : os.PathLike
        Path to the newly created sbi directory.
    """
    sbidir = make_unique_dir(rescalerdir, prefix)

    source = EXAMPLE_CONFIGS_DIR/SBI_CONFIG_FILENAME
    destination = (sbidir/SBI_CONFIG_FILENAME).resolve()
    shutil.copyfile(source, destination)

    print(f'Created a new sbi config file at {destination}.',
          'Edit it as needed.')
    return sbidir


def setup_unfolderdir(rescalerdir, prefix='unfolder_'):
    """
    Setup an unfolder directory with an example unfolder_config.py file.

    Parameters
    ----------
    rescalerdir : os.PathLike
        Path in which to create the unfolder directory ``unfolderdir``.

    prefix : str
        ``unfolderdir`` will be named as the prefix followed by a
        number, to make it unique.

    Returns
    -------
    unfolderdir : os.PathLike
        Path to the newly created unfolder directory.
    """
    unfolderdir = make_unique_dir(rescalerdir, prefix)

    source = EXAMPLE_CONFIGS_DIR/UNFOLDER_CONFIG_FILENAME
    destination = (unfolderdir/UNFOLDER_CONFIG_FILENAME).resolve()
    shutil.copyfile(source, destination)

    print(f'Created a new unfolder config file at {destination}.',
          'Edit it as needed.')
    return unfolderdir


def get_summary(datadir, apply_mask=True):
    """
    Return DataFrame with injection parameters, SNR, and parameters in
    the target space of the normalizing flow.

    Parameters
    ----------
    datadir : os.PathLike
        Path to the run directory in which training or test data have
        been created.

    apply_mask : bool
        Whether to apply the boolean mask to the data.
    """
    datadir = Path(datadir)
    config = load_data_config(datadir.parent)

    # Injection parameters
    summary = pd.read_feather(datadir/PARAMETERS_FILENAME)

    # Add SNR
    with h5py.File(datadir/PREPROCESSED_DATA_FILENAME, "r"
                  ) as preprocessed_data:
        for key in 'd_h', 'h_h', 'h0_h0':
            summary[key] = np.sum(preprocessed_data[key], axis=1)

    summary['snr'] = summary['d_h'] / np.sqrt(summary['h_h'])
    summary['snr0'] = np.sqrt(summary['h0_h0'])

    # Add transformed parameters
    columns = list(config.TRANSFORM_CLASS.sampled_params)
    for par in config.TRANSFORM_CLASS.folded_params:
        columns[columns.index(par)] = f'folded_{par}'

    with h5py.File(datadir/FOLDED_SAMPLED_PARAMETERS_FILENAME, "r") as h5file:
        folded_sampled_parameters = pd.DataFrame(
            h5file["dataset"], columns=columns)

    cogwheel.utils.update_dataframe(summary, folded_sampled_parameters)

    # Apply mask
    if apply_mask:
        mask = np.load(datadir/MASK_FILENAME)
        summary = summary[mask]

    return summary


def get_preprocessed_data(datadir, apply_mask=True,
                          slice_=slice(None)) -> dict:
    """
    Load ``preprocessed_data`` and apply the ``mask`` to it.

    Parameters
    ----------
    datadir : os.PathLike
        Path to the run directory in which training or test data have
        been created.

    apply_mask : bool
        Whether to apply the mask in {datadir}/{MASK_FILENAME} to the
        loaded arrays.

    slice_ : slice
        Only load a slice of the data to preserve memory. The slice is
        applied before the mask.

    Returns
    -------
    dict
        keys match those of ``preprocessed_data``.
    """
    mask = None
    if apply_mask:
        mask = np.load(datadir/MASK_FILENAME)[slice_]

    preprocessed_data = {}
    with h5py.File(datadir/PREPROCESSED_DATA_FILENAME) as file:
        for key, arr in file.items():
            if key == 'fbin':
                preprocessed_data[key] = arr[:]
            elif not apply_mask:
                preprocessed_data[key] = arr[slice_]
            else:
                preprocessed_data[key] = arr[slice_][mask]

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

    Returns
    -------
    results : list
        ``[func(*args) for args in iterable]``.

    stats : pstats.Stats
        Profiling statistics.
    """
    if processes is None:
        processes = os.cpu_count()
    elif processes < 0:
        processes += os.cpu_count()
    else:
        processes = min(os.cpu_count(), processes)

    with tempfile.TemporaryDirectory() as profile_dir:
        profiled_func = functools.partial(_aux_profiled_func,
                                          func=func, profile_dir=profile_dir)

        with multiprocessing.Pool(processes, _worker_initializer) as pool:
            results = pool.map(profiled_func, iterable)

        # Aggregate the stats
        paths = map(str, Path(profile_dir).glob('*.prof'))
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


def get_best_device(by='utilization'):
    """
    Get the GPU with the best utilization or memory usage.

    If there are no GPUs available, return the CPU.
    If `nvidia-smi` is not available, return the default GPU.

    Parameters
    ----------
    by : str
        Either 'utilization' or 'memory'.

    Returns
    -------
    torch.device
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')

    if not shutil.which('nvidia-smi'):
        return torch.device('cuda')

    def query_gpu(query):
        result = subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={query}',
             '--format=csv,noheader,nounits'],
            encoding='utf-8')
        return [int(x) for x in result.strip().split('\n')]

    memory = query_gpu('memory.free')
    utilization = query_gpu('utilization.gpu')

    if by == 'utilization':
        def key(i):
            return utilization[i], -memory[i]
    elif by == 'memory':
        def key(i):
            return -memory[i], utilization[i]
    else:
        raise ValueError("`by` should be 'utilization' or 'memory'.")

    gpu_id = min(range(len(memory)), key=key)
    return torch.device(f'cuda:{gpu_id}')
