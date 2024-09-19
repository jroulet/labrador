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
import os
from pathlib import Path
import numpy as np
import pandas as pd

from cogwheel import data
from cogwheel import gw_utils
from cogwheel import waveform
import cogwheel.utils

from . import semicoherent_likelihood
from .waveform_model import PhenomenologicalWaveformGenerator
from . import utils


def simulate_and_preprocess_sample(simulator, data_preprocessor,
                                   parameters, transform_class,
                                   transform_dic):
    """
    Generate a signal based on parameters, add a noise realization,
    find a reference waveform and preprocess the data by heterodyning.

    Return
    ------
    preprocessed_data: dict
        Contains the following entries
            * heterodyned_data: complex array of shape (n_det, n_freq)
            * heterodyned_signal: complex array of shape (n_det, n_freq)
            * fbin: float array of shape (n_freq,)
            * coef: 1-d float array
            * processed_coef: 1-d float array

    folded_sampled_params: float array of shape (n_params,)
        Signal parameters expressed in the folded target space.

    unfolding_label: int
        Index of the region that the parameters belong to before
        applying folding. Takes a value between [0, 2**n_folded_params).
    """
    simulated_input = simulator.generate_data_and_reference_waveform(
        parameters)

    preprocessed_data, transform_kwargs = data_preprocessor.preprocess_data(
        **simulated_input)

    folded_sampled_params, unfolding_label = _get_folded_sampled_params(
        parameters, transform_class, transform_kwargs, transform_dic)

    return preprocessed_data, folded_sampled_params, unfolding_label


def get_transform_dic(config):
    """
    Return a dictionary with transform kwargs that are the same across
    simulations.
    """
    return {key: config.PRIOR_KWARGS[key]
            for key in ('detector_pair', 'tgps', 'ref_det_name', 'f_avg',
                        'q_min')}


def get_i_refdet(config):
    """Return index of the reference detector."""
    return config.EVENT_DATA_KWARGS['detector_names'].index(
        config.PRIOR_KWARGS['ref_det_name'])


def _get_folded_sampled_params(parameters, transform_class,
                               transform_kwargs, transform_dic):
    """
    Return
    ------
    folded_sampled_params: float array of shape (n_params,)
        Signal parameters expressed in the folded target space.

    unfolding_label: int
        Index of the region that the parameters belong to before
        applying folding. Takes a value between [0, 2**n_folded_params).
    """
    transform = transform_class(**transform_dic, **transform_kwargs)
    sampled_params = transform.inverse_transform(
        **parameters[transform.standard_params])

    # Determine which region the truth would be unfolded to:
    folded_inds = transform._folded_inds
    values = np.fromiter(sampled_params.values(), float)[folded_inds]
    midpoint = (transform.cubemin[folded_inds]
                + transform.folded_cubesize[folded_inds])
    flags = values > midpoint
    # Convert the array of booleans to an integer
    unfolding_label = sum(val << i for i, val in enumerate(flags))

    folded_sampled_params = transform.fold(**sampled_params)
    return folded_sampled_params, unfolding_label


def simulate_and_preprocess_samples(simulator,
                                    data_preprocessor,
                                    simulation_parameters,
                                    transform_class,
                                    transform_dic,
                                    processes):
    """
    Run ``simulate_and_preprocess_sample()`` on a set of simulation
    parameter samples in parallel using ``multiprocessing``.

    Note: For best results you may want to ensure that each process
    runs a single thread, by running
    ```
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    ```
    at the very start of your Python session (in particular, before
    importing ``numpy`` or any module that imports it).

    Parameters
    ----------
    simulator: Simulator

    data_preprocessor: DataPreprocessor

    simulation_parameters: pandas.DataFrame
        Columns represent different parameters, each row is a
        simulation. The columns must contain all
        ``simulator._waveform_generator.params``.

    transform_dic: dict
        Transform kwargs that are the same across simulations. See
        ``get_transform_dic``.

    processes: int or None
        The number of worker processes to use. If `processes` is
        `None` then the number returned by `os.cpu_count()` is used.

    Return
    ------
    preprocessed_data: dict
        Contains the following entries
            * heterodyned_data: (n_sim, n_det, n_freq) complex array
            * heterodyned_signal: (n_sim, n_det, n_freq) complex array
            * fbin: (n_freq,) float array
            * coef: (n_sim, n_coef) float array
            * processed_coef: (n_sim, n_processed_coef) float array

    folded_sampled_params: (n_sim, n_params) float32 array
            Signal parameters expressed in the folded target space.

    unfolding_labels: (n_sim,) int array
        Index of the region that the parameters of each simulation
        belong to before applying folding. Takes values between
        [0, 2**n_folded_params).
    """
    args_generator = ((simulator, data_preprocessor, parameters,
                       transform_class, transform_dic)
                      for _, parameters in simulation_parameters.iterrows())
    results, stats = utils.multiprocessing_starmap_profiled(
        simulate_and_preprocess_sample, args_generator, processes)

    preprocessed_rows, folded_sampled_params, unfolding_labels = zip(*results)
    del results

    # Turn tuple of dict into dict of arrays

    # fbin should be identical across simulations, keep only one:
    preprocessed_data = {'fbin': preprocessed_rows[0]['fbin']}
    for row in preprocessed_rows:
        del row['fbin']

    for key, arr in preprocessed_rows[0].copy().items():
        preprocessed_data[key] = np.fromiter(
            (row.pop(key) for row in preprocessed_rows),
            dtype=(arr.dtype, arr.shape),
            count=len(preprocessed_rows))

    return (preprocessed_data,
            np.array(folded_sampled_params, np.float32),
            np.array(unfolding_labels),
            stats)


class Simulator:
    """
    Methods for generating data similar to what a user would provide.
    """

    def __init__(self, event_data_kwargs, approximant):
        """
        Parameters
        ----------
        event_data_kwargs: dict
            Keyword arguments to cogwheel.data.EventData.gaussian_noise

        approximant: str
            Name of the approximant used to inject a signal.
        """
        self.event_data_kwargs = event_data_kwargs
        self.approximant = approximant

        dummy_event_data = data.EventData.gaussian_noise(
            **self.event_data_kwargs)
        self._waveform_generator = waveform.WaveformGenerator.from_event_data(
            dummy_event_data, approximant)

    def generate_data_and_reference_waveform(self, parameters):
        """
        Generate data similar to what a user would provide.

        Parameters
        ----------
        parameters: dict-like
            Physical parameters of the signal to simulate. Must contain
            keys for all ``._waveform_generator.params``.

        Return
        ------
        dict: Contains the following entries
            * event_data
            * frequencies
            * ref_waveform_amp
            * ref_waveform_phase
            These can be passed to ``DataPreprocessor.preprocess_data``.
        """
        event_data = data.EventData.gaussian_noise(**self.event_data_kwargs)
        event_data.inject_signal(parameters, self.approximant)
        frequencies = event_data.frequencies[event_data.fslice]

        # Cheating: user "knows" true parameters. TODO improve this?
        # Although in principle our likelihood maximization erases this...
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

    def __init__(self,
                 waveform_model,
                 i_refdet,
                 f_ref,
                 n_coherent_segments=8,
                 pn_phase_tol_compression=None):
        """
        Parameters
        ----------
        waveform_model: waveform_model.PhenomenologicalWaveformGenerator
            Used to generate the reference waveform.

        n_coherent_segments: int
            When maximizing the likelihood to find a reference waveform,
            the frequency range is partitioned into segments and a
            constant phase is optimized independently in each segment.
            This is unphysical and intended to make the maximization
            more robust to limitations in the phase model.
            ``n_coherent_segments=1`` corresponds to fully coherent.

        pn_phase_tol_compression: float
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
        event_data: cogwheel.data.EventData
            Data containing the event.

        frequencies: float array of shape (n_freq,)
            Frequency array on which the user's reference waveform is
            defined. For now, it must match
            ``event_data.frequencies[event_data.fslice]``.

        ref_waveform_amp: float array of shape (n_det, n_freq)
            User-provided reference waveform amplitude.

        ref_waveform_phase: float array of shape (n_det, n_freq)
            User-provided reference waveform unwrapped phase.

        Return
        ------
        preprocessed_data: dict
            Contains the following entries
                * heterodyned_data
                * heterodyned_signal
                * fbin
                * coef
                * processed_coef

        transform_kwargs: dict
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

        like = semicoherent_likelihood.SemicoherentLikelihood(
            event_data=event_data,
            ref_waveform_phase=ref_waveform_phase,
            waveform_model=self.waveform_model,
            n_coherent_segments=self.n_coherent_segments)

        coef, d_h0_semicoherent, h0_h0 = like.fit_coef(
            frequencies,
            ref_waveform_phase=ref_waveform_phase,
            ref_waveform_amp=ref_waveform_amp)

        heterodyned_data, heterodyned_signal, fbin \
            = like.get_heterodyned_data_and_signal(
                coef, self.pn_phase_tol_compression)

        processed_coef = self.waveform_model.process_coef(coef, self.i_refdet)

        preprocessed_data = {
            'heterodyned_data': heterodyned_data,
            'heterodyned_signal': heterodyned_signal,
            'fbin': fbin,
            'coef': coef,
            'processed_coef': processed_coef,
            'd_h0_semicoherent': d_h0_semicoherent,
            'h0_h0': h0_h0,
            'd_h': event_data.injection['d_h'],
            'h_h': event_data.injection['h_h']}

        return preprocessed_data


def _check_rundir(rundir):
    utils.check_version(rundir)

    datadirs = rundir/utils.TRAINING_DIR, rundir/utils.TEST_DIR

    new_filenames = (utils.PREPROCESSED_DATA_FILENAME,
                     utils.FOLDED_SAMPLED_PARAMS_FILENAME,
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


def submit_condor(rundir,
                  request_cpus,
                  request_memory='5G',
                  request_disk='1G',
                  **submit_kwargs):
    """
    Submit an HTCondor job to simulate training data.

    This will generate the following files:
        {submission_scripts}/simulation.{sub,sh,out,err,log}

    Parameters
    ----------
    rundir: str, os.PathLike
        Run directory, should contain a file `data_config.py` and
        training and test directories with simulation parameters.

    request_cpus, request_memory, request_disk: int or str
        Specifications in the HTCondor submit file.

    **submit_kwargs
        Further options to include in the HTCondor submit file. Do
        not pass `executable`, `output`, `error`, `log`, `args`,
        `queue`, which will be dealt with automatically.
    """
    rundir = Path(rundir).resolve()
    _check_rundir(rundir)
    scripts_dir = rundir/'submission_scripts'
    os.makedirs(scripts_dir, exist_ok=True)

    submit_kwargs = {
        'submit_path': scripts_dir/'simulation.sub',
        'executable': scripts_dir/'simulation.sh',
        'output': scripts_dir/'simulation.out',
        'error': scripts_dir/'simulation.err',
        'log': scripts_dir/'simulation.log',
        'args': f'{rundir} --processes {request_cpus}',
        'request_cpus': request_cpus,
        'request_memory': request_memory,
        'request_disk': request_disk,
        } | submit_kwargs

    cogwheel.utils.submit_condor(**submit_kwargs)


def _populate_datadir(datadir, simulator, data_preprocessor,
                      transform_class, transform_dic, processes):
    simulation_parameters = pd.read_feather(datadir/utils.PARAMETERS_FILENAME)
    preprocessed_data, folded_sampled_params, unfolding_labels, stats \
        = simulate_and_preprocess_samples(
            simulator,
            data_preprocessor,
            simulation_parameters,
            transform_class=transform_class,
            transform_dic=transform_dic,
            processes=processes)

    np.savez(datadir/utils.PREPROCESSED_DATA_FILENAME, **preprocessed_data)
    np.save(datadir/utils.FOLDED_SAMPLED_PARAMS_FILENAME,
            folded_sampled_params)
    np.save(datadir/utils.UNFOLDING_LABELS_FILENAME, unfolding_labels)
    stats.dump_stats(datadir/'simulation_profiling')


def setup_simulator(rundir):
    """
    Parameters
    ----------
    rundir: str, os.PathLike
        Run directory, should contain a file `data_config.py`.

    Returns
    -------
    simulator: Simulator

    data_preprocessor: DataPreprocessor

    transform_class: type
        Read from {rundir}/config.py

    transform_dic: dict
        Event-independent kwargs to ``transform_class``.
    """
    rundir = Path(rundir)
    _check_rundir(rundir)

    config = utils.load_data_config(rundir)

    simulator = Simulator(config.EVENT_DATA_KWARGS, config.APPROXIMANT)

    waveform_model = PhenomenologicalWaveformGenerator.from_rundir(rundir)

    data_preprocessor = DataPreprocessor(
        waveform_model,
        i_refdet=get_i_refdet(config),
        f_ref=config.PRIOR_KWARGS['f_ref'],
        pn_phase_tol_compression=config.PN_PHASE_TOL_COMPRESSION,
        n_coherent_segments=config.N_COHERENT_SEGMENTS)

    transform_class = config.TRANSFORM_CLASS
    transform_dic = get_transform_dic(config)
    return simulator, data_preprocessor, transform_class, transform_dic


def main(rundir, processes=None):
    """Generate and preprocess training and test data."""
    simulator, data_preprocessor, transform_class, transform_dic \
        = setup_simulator(rundir)

    for dirname in utils.TRAINING_DIR, utils.TEST_DIR:
        _populate_datadir(rundir/dirname, simulator, data_preprocessor,
                          transform_class, transform_dic, processes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate signals to generate training and test data.')
    parser.add_argument('rundir',
                        help='''Simulation directory path, on which
                                `generate_parameters` has already been run.''')

    parser.add_argument('--processes', type=int, help='Number of processes')
    main(**vars(parser.parse_args()))
