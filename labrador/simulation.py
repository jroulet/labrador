"""
This module defines the classes Simulator and DataPreprocessor, and can
be run as a script to produce training data.

Classes
-------
Simulator:
    Generate data similar to user input.

DataPreprocessor:
    Compress data by heterodyning against a reference waveform.
"""
import argparse
import csv
import functools
import os
from pathlib import Path
import sys
import textwrap

import h5py
import pyarrow.feather
import numpy as np
import pandas as pd

from cogwheel import data, gw_utils, waveform

from . import condor_utils, semicoherent_likelihood, utils
from .waveform_model import PhenomenologicalWaveformGenerator


def setup_simulator(rundir):
    """
    Parameters
    ----------
    rundir : str, os.PathLike
        Run directory, should contain a file `data_config.py`.

    Returns
    -------
    simulator : Simulator

    data_preprocessor : DataPreprocessor

    transform_class : type
        Read from {rundir}/data_config.py
    """
    rundir = Path(rundir)
    config = utils.load_data_config(rundir)

    simulator = Simulator(config.EVENT_DATA_KWARGS, config.APPROXIMANT)
    data_preprocessor = DataPreprocessor.from_rundir(rundir)
    transform_class = _get_transform_class(config)
    return simulator, data_preprocessor, transform_class


def simulate_and_preprocess_sample(simulator, data_preprocessor,
                                   parameters, transform_class,
                                   seed=None):
    """
    Generate a signal based on parameters, add a noise realization, find
    a reference waveform and preprocess the data by heterodyning.

    Returns
    -------
    preprocessed_data : dict
        Contains the following entries

            * heterodyned_data: complex array of shape (n_det, n_freq)
            * heterodyned_signal: complex array of shape (n_det, n_freq)
            * fbin: float array of shape (n_freq,)
            * coef: 1-d float array
            * processed_coef: 1-d float array

    folded_sampled_parameters : float array of shape (n_parameters,)
        Signal parameters expressed in the folded target space.

    unfolding_label : int
        Index of the region that the parameters belong to before
        applying folding. Takes a value between
        [0, 2**n_folded_parameters).

    seed : int
        Determines the Gaussian noise realization of the simulated data.
    """
    simulated_input = simulator.generate_data_and_reference_waveform(
        parameters, seed)

    preprocessed_data, transform_kwargs = data_preprocessor.preprocess_data(
        **simulated_input)

    transform = transform_class(**transform_kwargs)
    folded_sampled_parameters, unfolding_label = get_folded_sampled_parameters(
        parameters, transform)

    return preprocessed_data, folded_sampled_parameters, unfolding_label


def _get_transform_class(config):
    """
    Return a transform class partially instantiatied with kwargs that
    are the same across simulations.
    """
    return functools.partial(config.TRANSFORM_CLASS, **config.PRIOR_KWARGS)


def _get_i_refdet(config):
    """Return index of the reference detector."""
    return config.EVENT_DATA_KWARGS['detector_names'].index(
        config.PRIOR_KWARGS['ref_det_name'])


def get_folded_sampled_parameters(parameters, transform):
    """
    Returns
    -------
    folded_sampled_parameters : float array of shape (n_parameters,)
        Signal parameters expressed in the folded target space.

    unfolding_label : int
        Index of the region that the parameters belong to before
        applying folding. Takes a value between [0, 2**n_folded_params).
    """
    sampled_parameters = transform.inverse_transform(
        **parameters[transform.standard_params])

    # Determine which region the truth would be unfolded to:
    folded_inds = transform._folded_inds
    values = np.fromiter(sampled_parameters.values(), float)[folded_inds]
    midpoint = (transform.cubemin[folded_inds]
                + transform.folded_cubesize[folded_inds])
    flags = values > midpoint
    # Convert the array of booleans to an integer
    unfolding_label = sum(val << i for i, val in enumerate(flags[::-1]))

    folded_sampled_parameters = transform.fold(**sampled_parameters)
    return folded_sampled_parameters, unfolding_label


def simulate_and_preprocess_samples(simulator,
                                    data_preprocessor,
                                    simulation_parameters,
                                    transform_class,
                                    processes,
                                    seed_suffix):
    """
    Run :py:func:`simulate_and_preprocess_sample` on a set of simulation
    parameter samples in parallel using ``multiprocessing``.

    Note: For best results you may want to ensure that each process runs
    a single thread, by running

    .. code-block:: python

        import os
        os.environ["OMP_NUM_THREADS"] = "1"

    at the very start of your Python session (in particular, before
    importing ``numpy`` or any module that imports it).

    Parameters
    ----------
    simulator : Simulator

    data_preprocessor : DataPreprocessor

    simulation_parameters : pandas.DataFrame
        Columns represent different parameters, each row is a
        simulation. The columns must contain all
        ``simulator._waveform_generator.params``. The index will be used
        as seed for the Gaussian noise.

    processes : int or None
        The number of worker processes to use. If `processes` is
        `None` then the number returned by `os.cpu_count()` is used.

    seed_suffix : e.g. bool
        Used to prevent the training and test sets from having the same
        noise realizations.

    Returns
    -------
    preprocessed_data : dict
        Contains the following entries:

            * heterodyned_data: (n_sim, n_det, n_freq) complex array
            * heterodyned_signal: (n_sim, n_det, n_freq) complex array
            * fbin: (n_freq,) float array
            * coef: (n_sim, n_coef) float array
            * processed_coef: (n_sim, n_processed_coef) float array

    folded_sampled_parameters : (n_sim, n_parameters) float32 array
        Signal parameters expressed in the folded target space.

    unfolding_labels : (n_sim,) int array
        Index of the region that the parameters of each simulation
        belong to before applying folding. Takes values between
        [0, 2**n_folded_parameters).

    cpu_time : float
        Total CPU time in seconds spent across all processes.
    """
    args_generator = (
        (
            simulator,
            data_preprocessor,
            parameters,
            transform_class,
            (seed_prefix, seed_suffix)
        )
        for seed_prefix, parameters in simulation_parameters.iterrows()
    )

    results, time = utils.multiprocessing_starmap_timed(
        simulate_and_preprocess_sample, args_generator, processes)

    preprocessed_rows, folded_sampled_parameters, unfolding_labels = zip(
        *results)
    del results

    # fbin should be identical across simulations, keep only one:
    preprocessed_data = {'fbin': preprocessed_rows[0]['fbin']}
    for row in preprocessed_rows:
        del row['fbin']

    # Turn tuple of dict into dict of arrays
    for key, arr in preprocessed_rows[0].copy().items():
        preprocessed_data[key] = np.fromiter(
            (row.pop(key) for row in preprocessed_rows),
            dtype=(arr.dtype, arr.shape),
            count=len(preprocessed_rows))

    return (preprocessed_data,
            np.array(folded_sampled_parameters, np.float32),
            np.array(unfolding_labels),
            time)


class Simulator:
    """
    Methods for generating data similar to what a user would provide.
    """

    def __init__(self, event_data_kwargs, approximant):
        """
        Parameters
        ----------
        event_data_kwargs : dict
            Keyword arguments to cogwheel.data.EventData.gaussian_noise

        approximant : str
            Name of the approximant used to inject a signal.
        """
        self.event_data_kwargs = event_data_kwargs
        self.approximant = approximant

        dummy_event_data = data.EventData.gaussian_noise(
            **self.event_data_kwargs)
        self._waveform_generator = waveform.WaveformGenerator.from_event_data(
            dummy_event_data, approximant)

    def generate_data_and_reference_waveform(self, parameters,
                                             seed=None):
        """
        Generate data similar to what a user would provide.

        Parameters
        ----------
        parameters : dict-like
            Physical parameters of the signal to simulate. Must contain
            keys for all ``._waveform_generator.params``.

        seed : int
            Determines the Gaussian noise realization of the simulated
            data.

        Returns
        -------
        dict : Contains the following entries:

            * event_data
            * frequencies
            * ref_waveform_amp
            * ref_waveform_phase

            These can be passed to ``DataPreprocessor.preprocess_data``.
        """
        event_data = data.EventData.gaussian_noise(
            **self.event_data_kwargs, seed=seed)
        event_data.inject_signal(parameters, self.approximant)
        frequencies = event_data.frequencies[event_data.fslice]

        # Cheating: knows true parameters, but likelihood maximization
        # erases this.
        signal = self._waveform_generator.get_strain_at_detectors(
            frequencies, parameters)
        mchirp = gw_utils.m1m2_to_mchirp(**parameters[['m1', 'm2']])
        ref_waveform_phase = semicoherent_likelihood.get_unwrapped_phase(
            frequencies, signal, mchirp)
        ref_waveform_amp = np.abs(signal)
        return {'event_data': event_data,
                'frequencies': frequencies,
                'ref_waveform_amp': ref_waveform_amp,
                'ref_waveform_phase': ref_waveform_phase}


class DataPreprocessor:
    """
    Methods for compressing the data by heterodyning against a
    phenomenological reference waveform.
    """

    @classmethod
    def from_rundir(cls, rundir):
        """
        Constructor from a run directory.

        Parameters
        ----------
        rundir : os.PathLike
            Run directory, should contain a file `data_config.py` and
            training and test directories with simulation parameters.
        """

        rundir = Path(rundir)
        config = utils.load_data_config(rundir)

        waveform_model = PhenomenologicalWaveformGenerator.from_rundir(rundir)
        return cls(waveform_model,
                   i_refdet=_get_i_refdet(config),
                   f_ref=config.PRIOR_KWARGS['f_ref'],
                   n_coherent_segments=config.N_COHERENT_SEGMENTS,
                   pn_phase_tol_compression=config.PN_PHASE_TOL_COMPRESSION)

    def __init__(self,
                 waveform_model,
                 i_refdet,
                 f_ref,
                 n_coherent_segments=8,
                 pn_phase_tol_compression=None):
        """
        Parameters
        ----------
        waveform_model : waveform_model.PhenomenologicalWaveformGenerator
            Used to generate the reference waveform.

        n_coherent_segments : int
            When maximizing the likelihood to find a reference waveform,
            the frequency range is partitioned into segments and a
            constant phase is optimized independently in each segment.
            This is unphysical and intended to make the maximization
            more robust to limitations in the phase model.
            ``n_coherent_segments=1`` corresponds to fully coherent.

        pn_phase_tol_compression : float
            Controls the relative-binning frequency resolution used for
            compressing the data after the reference waveform has been
            found. Lower tolerance means higher resolution.
        """
        self.waveform_model = waveform_model
        self.n_coherent_segments = n_coherent_segments
        self.pn_phase_tol_compression = pn_phase_tol_compression
        self.i_refdet = i_refdet
        self.f_ref = f_ref

    def preprocess_data(self,
                        event_data,
                        frequencies,
                        ref_waveform_amp,
                        ref_waveform_phase):
        """
        Compress the data by heterodyning it against a phenomenological
        reference waveform.

        The phenomenological reference waveform is found by first
        fitting a reference provided by the user, and then optimizing a
        semi-coherent likelihood using that as initial guess.
        The purpose of this optimization is to be insensitive to how the
        user found their reference waveform: we cannot control this and
        so we cannot trust that the training will capture it.

        Parameters
        ----------
        event_data : cogwheel.data.EventData
            Data containing the event.

        frequencies : float array of shape (n_freq,)
            Frequency array on which the user's reference waveform is
            defined. For now, it must match
            ``event_data.frequencies[event_data.fslice]``.

        ref_waveform_amp : float array of shape (n_det, n_freq)
            User-provided reference waveform amplitude.

        ref_waveform_phase : float array of shape (n_det, n_freq)
            User-provided reference waveform unwrapped phase.

        Returns
        -------
        preprocessed_data : dict
            Contains the following entries:

            * heterodyned_data
            * fbin
            * coef
            * processed_coef
            * h0_h0

            Plus, only if `event_data` is an injection:

            * heterodyned_signal
            * d_h
            * h_h

        transform_kwargs : dict
            Contains event-dependent keyword arguments to the target-
            space coordinate transform.
        """
        preprocessed_data = self._fit_waveform_and_heterodyne_data(
            event_data, frequencies, ref_waveform_amp, ref_waveform_phase)

        transform_kwargs = self.waveform_model.get_transform_kwargs(
            preprocessed_data['coef'], self.i_refdet, self.f_ref)

        return preprocessed_data, transform_kwargs

    def _fit_waveform_and_heterodyne_data(self,
                                          event_data,
                                          frequencies,
                                          ref_waveform_amp,
                                          ref_waveform_phase):
        # TODO generalize frequencies
        assert np.array_equal(frequencies,
                              event_data.frequencies[event_data.fslice])

        rb_splines = self.waveform_model.phase_model.rb_splines
        if not np.array_equal(frequencies, rb_splines.frequencies):
            print(f'Changing {self.__class__.__name__} frequency grid '
                  'to match that of `event_data`.')
            self.waveform_model.phase_model.rb_splines \
                = rb_splines.reinstantiate(frequencies=frequencies,
                                           pn_phase_tol=None,
                                           fbin=rb_splines.fbin)

        like = semicoherent_likelihood.SemicoherentLikelihood(
            event_data=event_data,
            waveform_model=self.waveform_model,
            n_coherent_segments=self.n_coherent_segments)

        coef, h0_h0 = like.fit_coef(frequencies,
                                    ref_waveform_phase=ref_waveform_phase,
                                    ref_waveform_amp=ref_waveform_amp)

        heterodyned_data, heterodyned_signal, fbin \
            = like.get_heterodyned_data_and_signal(
                coef, self.pn_phase_tol_compression)

        processed_coef = self.waveform_model.process_coef(coef, self.i_refdet)

        preprocessed_data = {
            'heterodyned_data': heterodyned_data,
            'fbin': fbin,
            'coef': coef,
            'processed_coef': processed_coef,
            'h0_h0': h0_h0,
        }

        if event_data.injection:
            preprocessed_data['heterodyned_signal'] = heterodyned_signal
            preprocessed_data['d_h'] = event_data.injection['d_h']
            preprocessed_data['h_h'] = event_data.injection['h_h']

        return preprocessed_data


def _check_rundir(rundir):
    try:
        utils.check_version(rundir)
    except FileNotFoundError as err:
        raise RuntimeError('Run `labrador.generate_parameters` first') from err

    datadirs = rundir/utils.TRAINING_DIR, rundir/utils.TEST_DIR

    new_filenames = (utils.PREPROCESSED_DATA_FILENAME,
                     utils.FOLDED_SAMPLED_PARAMETERS_FILENAME,
                     utils.UNFOLDING_LABELS_FILENAME)

    # Check that there is no data already
    existing = [path for filename in new_filenames for datadir in datadirs
                if (path := datadir/filename).exists()]
    if existing:
        raise FileExistsError(f'{existing} already exist!')

    # Check that data_config file exists
    data_config_file = rundir/utils.DATA_CONFIG_FILENAME
    if not data_config_file.exists():
        raise FileNotFoundError(f'Missing {data_config_file}')

    # Check that simulation parameters have already been generated
    for datadir in datadirs:
        parameters_file = datadir/utils.PARAMETERS_FILENAME
        if not parameters_file.exists():
            raise FileNotFoundError(
                f'Missing {parameters_file}, run `generate_parameters.py`.')


# ----------------------------------------------------------------------
# Chunking functions

CHUNKS_DIRNAME = 'chunks'
CHUNKS_FILENAME = 'chunks.csv'
PROFILE_FILENAME = 'simulation_time.npy'


def _setup_chunks(rundir, chunk_size):
    rundir = Path(rundir)
    data_config = utils.load_data_config(rundir)

    dirname_nsamples = [
        (utils.TEST_DIR, data_config.N_TEST_SIMULATIONS),
        (utils.TRAINING_DIR, data_config.N_TRAINING_SIMULATIONS),
    ]

    for dirname, n_samples in dirname_nsamples:
        chunksdir = rundir/dirname/CHUNKS_DIRNAME
        filepath = chunksdir/CHUNKS_FILENAME

        ind_pairs = [(i_start, min(i_start + chunk_size, n_samples))
                     for i_start in range(0, n_samples, chunk_size)]

        os.makedirs(chunksdir)
        with open(filepath, 'w', newline='', encoding='utf-8') as file:
            csv.writer(file).writerows(ind_pairs)


def _validate_chunkpaths(chunkpaths):
    # Validate that all chunks are there
    ind_pairs = [_get_chunk_start_and_end(path) for path in chunkpaths]
    for current, following in zip(ind_pairs, ind_pairs[1:]):
        if current[1] != following[0]:
            raise RuntimeError(
                'Missing chunk(s) between {current}, {following}')
    assert ind_pairs[0][0] == 0


def simulate_chunk(datadir, i_start, i_end, processes):
    """
    Simulate data for a chunk of simulation parameters.

    Parameters
    ----------
    datadir : os.PathLike
        Path to the directory containing the simulation parameters.

    i_start, i_end : int
        Start (inclusive) and end (exclusive) indices for the chunk.

    processes : int
        Number of processes to use for parallelization.

    See Also
    --------
    cli.htcondor
        Orchestrates the generation of parameters, simulation in chunks
        and merging with HTCondor.

    cli.simulation_chunks
        Defines a command-line interface to this function.
    """
    datadir = Path(datadir).resolve()
    seed_suffix = datadir.name == utils.TRAINING_DIR
    rundir = datadir.parent
    chunksdir = datadir/CHUNKS_DIRNAME
    if not chunksdir.exists():
        raise FileNotFoundError(f'{chunksdir} missing, set up chunks first!')

    names = (
        utils.PREPROCESSED_DATA_FILENAME,
        utils.FOLDED_SAMPLED_PARAMETERS_FILENAME,
        utils.UNFOLDING_LABELS_FILENAME,
    )
    chunkpaths = [_get_chunkpath(chunksdir, name, i_start, i_end)
                  for name in names]

    if all(path.exists() for path in chunkpaths):
        print(f'Skipping existing chunk {i_start}..{i_end}')
        return

    simulator, data_preprocessor, transform_class = setup_simulator(rundir)

    parameters_chunk = _load_chunk_from_feather(
        datadir/utils.PARAMETERS_FILENAME, i_start, i_end)

    (
        preprocessed_data,
        folded_sampled_parameters,
        unfolding_labels,
        chunk_time,
    ) = simulate_and_preprocess_samples(simulator,
                                        data_preprocessor,
                                        parameters_chunk,
                                        transform_class=transform_class,
                                        processes=processes,
                                        seed_suffix=seed_suffix)

    datasets = (
        preprocessed_data,
        {'dataset': folded_sampled_parameters},
        {'dataset': unfolding_labels},
    )  # Order must be the same as that of `chunkpaths`

    for path, dataset in zip(chunkpaths, datasets):
        with h5py.File(path, 'w') as file:
            for key, arr in dataset.items():
                file.create_dataset(key, data=arr)

    np.save(_get_chunkpath(chunksdir, PROFILE_FILENAME, i_start, i_end),
            chunk_time)


def merge_chunks(rundir, delete_chunks_after_merging=True):
    """
    Merge chunks of simulations into single files.

    Parameters
    ----------
    rundir : os.PathLike
        Run directory; chunks of simulations should have been completed
        by the time this function is run.

    delete_chunks_after_merging : bool
        If True, delete the chunk files after merging.

    See Also
    --------
    simulate_chunk
    """
    rundir = Path(rundir)
    for dirname in utils.TEST_DIR, utils.TRAINING_DIR:
        _merge_chunks_in_datadir(rundir/dirname, delete_chunks_after_merging)


def _merge_chunks_in_datadir(datadir, delete_chunks_after_merging):
    chunksdir = datadir/CHUNKS_DIRNAME
    all_chunkpaths = []

    # HDF5 files:
    names = (
        utils.PREPROCESSED_DATA_FILENAME,
        utils.FOLDED_SAMPLED_PARAMETERS_FILENAME,
        utils.UNFOLDING_LABELS_FILENAME,
    )
    for name in names:
        pattern = _get_chunkpath('', name, '*', '*').name
        chunkpaths = sorted(chunksdir.glob(pattern),
                            key=_get_chunk_start_and_end)

        _validate_chunkpaths(chunkpaths)

        _, n_rows = _get_chunk_start_and_end(chunkpaths[-1])

        # Create the merged output file:
        with h5py.File(datadir/name, 'w') as merged:
            # Create datasets:
            with h5py.File(chunkpaths[0], 'r') as sample_file:
                for key, arr in sample_file.items():
                    if key == 'fbin':
                        merged.create_dataset(key, data=arr)
                    else:
                        merged.create_dataset(key,
                                              shape=(n_rows, *arr.shape[1:]),
                                              dtype=arr.dtype)

            # Populate datasets:
            for chunkpath in chunkpaths:
                i_start, i_end = _get_chunk_start_and_end(chunkpath)
                with h5py.File(chunkpath, 'r') as f_in:
                    for key, arr in f_in.items():
                        if key == 'fbin':
                            np.testing.assert_array_equal(merged[key], arr)
                        else:
                            merged[key][i_start : i_end] = arr

        all_chunkpaths.extend(chunkpaths)

    # Profiling statistics:
    pattern = _get_chunkpath('', PROFILE_FILENAME, '*', '*').name
    chunkpaths = list(chunksdir.glob(pattern))

    # pstats.Stats(*map(str, chunkpaths)).dump_stats(datadir/PROFILE_FILENAME)
    np.save(datadir/PROFILE_FILENAME, sum(map(np.load, chunkpaths)))

    all_chunkpaths.extend(chunkpaths)

    # Delete chunk files:
    if delete_chunks_after_merging:
        for chunkpath in all_chunkpaths:
            chunkpath.unlink()
        (chunksdir/CHUNKS_FILENAME).unlink()
        # Delete the chunks directory if empty:
        if not any(chunksdir.iterdir()):
            chunksdir.rmdir()

    print(f'Merged {len(all_chunkpaths)} chunk files into {datadir}.')


def _get_chunkpath(chunksdir, name, i_start, i_end):
    auxpath = Path(chunksdir)/name
    return auxpath.with_stem(f'{auxpath.stem}-{i_start}_{i_end}')


def _get_chunk_start_and_end(chunkpath) -> tuple[int, int]:
    i_start, i_end = map(int, chunkpath.stem.split('-')[-1].split('_'))
    return i_start, i_end


def _load_chunk_from_feather(feather_path: str, i_start: int,
                             i_end: int) -> pd.DataFrame:
    table = pyarrow.feather.read_table(feather_path, memory_map=True)
    chunk = table.slice(offset=i_start, length=i_end-i_start)
    return chunk.to_pandas().set_index(pd.RangeIndex(i_start, i_end))


# ----------------------------------------------------------------------
# HTCondor functions

def setup_condor_sub(rundir, chunk_size,
                     delete_chunks_after_merging=True,
                     **submit_kwargs):
    """
    Set up HTCondor submission scripts to run multiple `simulate_chunk`
    and a `merge_chunks`.

    Parameters
    ----------
    rundir : os.PathLike
        Run directory, should contain a file `data_config.py` and
        training and test directories with simulation parameters.

    delete_chunks_after_merging : bool
        If True, delete the chunk files after merging.

    Returns
    -------
    submit_chunks_paths: tuple [pathlib.Path, pathlib.Path]
        Paths to the HTCondor submission scripts for simulating chunks
        for the training and test sets, respectively.

    submit_merge_path : pathlib.Path
        Path to the HTCondor submission scripts for mergining chunks.

    See Also
    --------
    cli.htcondor : Command-line interface to run this and other jobs.
    """
    rundir = Path(rundir).resolve()
    _setup_chunks(rundir, chunk_size)
    submit_chunks_paths = _setup_condor_for_simulate_chunks(
        rundir, **submit_kwargs)
    submit_merge_path = _setup_condor_for_merge_chunks(
        rundir, delete_chunks_after_merging, **submit_kwargs)
    return submit_chunks_paths, submit_merge_path


def _setup_condor_for_simulate_chunks(rundir,
                                      request_memory='8G',
                                      request_disk='1G',
                                      **submit_kwargs):
    """
    Set up HTCondor submission scripts to run `simulate_chunk` on
    training and test data chunks.

    Parameters
    ----------
    rundir : os.PathLike
        Run directory, should contain training and test directories
        with simulation parameters.

    Returns
    -------
    tuple of pathlib.Path
        Paths to the training and test HTCondor submission scripts.
    """
    scripts_dir = rundir/'submission_scripts'
    logdir = scripts_dir/'simulation_logs'
    os.makedirs(logdir, exist_ok=True)

    traindir = rundir/utils.TRAINING_DIR
    testdir = rundir/utils.TEST_DIR
    train_chunkfile = traindir/CHUNKS_DIRNAME/CHUNKS_FILENAME
    test_chunkfile = testdir/CHUNKS_DIRNAME/CHUNKS_FILENAME

    env_lib = Path(sys.executable).resolve().parents[1]/'lib'
    executable_path = scripts_dir/'simulation.sh'

    # Write executable (common to training and test sets):
    executable_text = textwrap.dedent(
        f"""\
        #!/bin/bash
        export OMP_NUM_THREADS=1
        export LD_LIBRARY_PATH="{env_lib}:$LD_LIBRARY_PATH"

        set -e

        {Path(sys.executable).resolve().parent/'lab-simulate-chunk'} "$@"
        """)

    condor_utils.write_executable(executable_path, executable_text)

    # Write separate submit files for the training and test sets:
    kwarg_lines = """
            """.join(f'{key} = {value}'
                     for key, value in submit_kwargs.items())

    submit_paths = []
    for kind, datadir, chunkfile in [('train', traindir, train_chunkfile),
                                     ('test', testdir, test_chunkfile)]:
        log_stem = logdir/f"simulation-$(i_min)_$(i_max)_{kind}"
        submit_path = scripts_dir/f"simulation_{kind}.sub"

        submit_text = textwrap.dedent(
            f"""\
            executable = {executable_path}
            request_memory = {request_memory}
            request_disk = {request_disk}

            {kwarg_lines}

            arguments = {datadir} $(i_min) $(i_max)
            output = {log_stem}.out
            error = {log_stem}.err
            log = {log_stem}.log
            queue i_min, i_max from {chunkfile}
            """)

        condor_utils.write_executable(submit_path, submit_text)
        submit_paths.append(submit_path)

    return tuple(submit_paths)


def _setup_condor_for_merge_chunks(rundir, delete_chunks_after_merging,
                                   request_disk='100G',
                                   request_memory='1G',
                                   **submit_kwargs):
    """
    Set up HTCondor submission script to run `merge_chunks` on
    chunks of data.

    Parameters
    ----------
    rundir : os.PathLike
        Run directory, chunks of simulations should have been completed
        by the time this function is run.

    delete_chunks_after_merging : bool
        If True, delete the chunk files after merging.

    Returns
    -------
    submit_path : pathlib.Path
        Path to the HTCondor submission script.
    """
    stem = rundir/'submission_scripts'/'merge_chunks'
    entry = 'lab-merge-chunks'
    arguments = str(rundir)
    if delete_chunks_after_merging:
        arguments += ' --delete_chunks_after_merging'

    submit_path = condor_utils.setup_condor_sub(
        stem, entry, arguments=arguments, request_disk=request_disk,
        request_memory=request_memory, **submit_kwargs)

    return submit_path


# ----------------------------------------------------------------------
# Functions to simulate data in the local computer

def main(rundir, processes=None):
    """Generate and preprocess training and test data."""
    rundir = Path(rundir)
    _check_rundir(rundir)

    simulator, data_preprocessor, transform_class = setup_simulator(rundir)

    for dirname in utils.TEST_DIR, utils.TRAINING_DIR:
        _populate_datadir(rundir/dirname, simulator, data_preprocessor,
                          transform_class, processes)


def _populate_datadir(datadir, simulator, data_preprocessor,
                      transform_class, processes, chunk_size=10_000):
    simulation_parameters = pd.read_feather(datadir/utils.PARAMETERS_FILENAME)
    seed_suffix = datadir.name == utils.TRAINING_DIR

    time = 0.0

    for chunk_start in range(0, len(simulation_parameters), chunk_size):
        (
            preprocessed_data,
            folded_sampled_parameters,
            unfolding_labels,
            chunk_time
        ) = simulate_and_preprocess_samples(
            simulator,
            data_preprocessor,
            simulation_parameters[chunk_start : chunk_start + chunk_size],
            transform_class=transform_class,
            processes=processes,
            seed_suffix=seed_suffix
        )

        _append_to_hdf5(datadir/utils.PREPROCESSED_DATA_FILENAME,
                        **preprocessed_data)
        _append_to_hdf5(datadir/utils.FOLDED_SAMPLED_PARAMETERS_FILENAME,
                        dataset=folded_sampled_parameters)
        _append_to_hdf5(datadir/utils.UNFOLDING_LABELS_FILENAME,
                        dataset=unfolding_labels)

        time += chunk_time

    np.save(datadir/PROFILE_FILENAME, time)


def _append_to_hdf5(filename, **arrays):
    """
    Append arrays to an hdf5 file.

    Parameters
    ----------
    filename : os.PathLike
        Path to an hdf5 file. If it doesn't exist, it will be created.

    **arrays
        Data to append. Keys are the groups in the hdf5.
    """
    with h5py.File(filename, "a") as h5file:
        for key, array in arrays.items():
            if key in h5file:
                # Resize along first axis and append new data:
                dataset = h5file[key]
                dataset.resize(dataset.shape[0] + array.shape[0], axis=0)
                dataset[-array.shape[0]:] = array
            else:
                h5file.create_dataset(key, data=array,
                                      maxshape=(None, *array.shape[1:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate signals to generate training and test data.')
    parser.add_argument('rundir',
                        help='''Simulation directory path, on which
                                `generate_parameters` has already been run.''')

    parser.add_argument('--processes', type=int, help='Number of processes')
    main(**vars(parser.parse_args()))
