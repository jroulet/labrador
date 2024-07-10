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
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd

from cogwheel import data
from cogwheel import gw_utils
from cogwheel import waveform
from cogwheel.validation import load_config
import cogwheel.utils

from . import semicoherent_likelihood
from .transform import TargetSpaceTransform
from .generate_parameters import PARAMETERS_FILENAME, CONFIG_FILENAME
from .waveform_model import PhenomenologicalWaveformGenerator


def simulate_and_preprocess_sample(simulator, data_preprocessor,
                                   parameters, transform_dic):
    """
    Generate a signal based on parameters, add a noise realization,
    find a reference waveform and compress the data by heterodyning.

    Return
    ------
    compressed_data: 1-d float32 array
        Features.
    """
    simulated_input = simulator.generate_data_and_reference_waveform(
        parameters)
    compressed_data, transform_kwargs, d_h0_semicoherent, h0_h0 \
        = data_preprocessor.preprocess_data(**simulated_input)
    folded_sampled_params, unfolding_label = _get_folded_sampled_params(
        parameters, transform_kwargs, transform_dic)

    arrs = {'d_h0_semicoherent': d_h0_semicoherent,
            'h0_h0': h0_h0,
            'd_h': simulated_input['event_data'].injection['d_h'],
            'h_h': simulated_input['event_data'].injection['h_h']}

    derived_parameters = {}
    for key, arr in arrs.items():
        derived_parameters[key] = arr.sum()
        for i, det in enumerate(simulated_input['event_data'].detector_names):
            derived_parameters[f'{key}_{det}'] = arr[i]

    return (compressed_data,
            folded_sampled_params,
            unfolding_label,
            derived_parameters)


def get_transform_dic(config):
    """
    Return a dictionary with transform kwargs that are the same across
    simulations.
    """
    return {key: config.PRIOR_KWARGS[key]
            for key in ('detector_pair', 'tgps', 'ref_det_name', 'f_avg')}


def get_i_refdet(config):
    """Return index of the reference detector."""
    return config.EVENT_DATA_KWARGS['detector_names'].index(
        config.PRIOR_KWARGS['ref_det_name'])


def _get_folded_sampled_params(parameters, transform_kwargs,
                               transform_dic):
    """
    Return
    ------
    folded_sampled_params: float array of shape (n_params,)
    unfolding_label: int
    """
    transform = TargetSpaceTransform(**transform_dic, **transform_kwargs)
    sampled_params = transform.inverse_transform(
        **parameters[transform.standard_params])

    # Determine which region the truth would be unfolded to:
    values = np.fromiter(sampled_params.values(),
                         float)[transform._folded_inds]
    midpoint = (transform.cubemin
                + transform.folded_cubesize)[transform._folded_inds]
    flags = values > midpoint
    # Convert the array of booleans to an integer
    unfolding_label = sum(val << i for i, val in enumerate(flags))

    folded_sampled_params = transform.fold(**sampled_params)
    return folded_sampled_params, unfolding_label


def simulate_and_preprocess_samples(simulator,
                                    data_preprocessor,
                                    simulation_parameters,
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
        ``._waveform_generator.params``.

    processes: int or None
        The number of worker processes to use. If `processes` is
        `None` then the number returned by `os.cpu_count()` is used.

    Return
    ------
    simulation_data: float32 array of shape (n_simulations, n_features)
    folded_sampled_params: float32 array of shape (n_simulations, n_params)
    unfolding_labels: int array of shape(n_simulations,)
    """
    with multiprocessing.Pool(processes) as pool:
        results = pool.starmap(
            simulate_and_preprocess_sample,
            ((simulator, data_preprocessor, parameters, transform_dic)
             for _, parameters in simulation_parameters.iterrows()))

    simulation_data, folded_sampled_params, unfolding_labels, derived = zip(
        *results)

    return (np.array(simulation_data, np.float32),
            np.array(folded_sampled_params, np.float32),
            np.array(unfolding_labels),
            pd.DataFrame.from_records(derived))


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
                 n_coherent_segments=8,
                 pn_phase_tol_compression=1.0):
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
        preprocessed_data: float array
            Contains the real and imaginary part of the heterodyned data
            at low frequency resolution, the parameters of the
            phenomenological reference waveform, and a few extra
            features that summarize the detector amplitude, phase and
            time differences.
        """
        # TODO generalize frequencies
        assert np.array_equal(frequencies,
                              event_data.frequencies[event_data.fslice])

        like = semicoherent_likelihood.SemicoherentLikelihood.from_event_data(
            event_data=event_data,
            ref_waveform_phase=ref_waveform_phase,
            waveform_model=self.waveform_model,
            n_coherent_segments=self.n_coherent_segments)

        coef, d_h0_semicoherent, h0_h0 = like.fit_coef(
            frequencies,
            ref_waveform_phase=ref_waveform_phase,
            ref_waveform_amp=ref_waveform_amp)

        heterodyned_data = self._get_heterodyned_data(like, coef)

        geometry_features = like.waveform_model.get_geometry_features(coef)

        preprocessed_data = np.concatenate([heterodyned_data.real.flat,
                                            heterodyned_data.imag.flat,
                                            coef,
                                            geometry_features])
        transform_kwargs = like.waveform_model.get_transform_kwargs(
            coef, self.i_refdet)

        return (preprocessed_data.astype(np.float32),
                transform_kwargs,
                d_h0_semicoherent,
                h0_h0)

    def _get_heterodyned_data(self, like, coef):
        """
        Parameters
        ----------
        like: SemicoherentLikelihood

        coef: float array
            Parameters of the best-fit phenomenological waveform, that
            will be used to heterodyne the data.

        Return
        ------
        heterodyned_data: complex array of shape (n_det, n_freq)
            Data, heterodyned with a reference waveform defined by
            `coef`. The frequency cutoff parameter is ignored in the
            reference waveform, to preserve high-frequency data.
            The amplitude is canceled out so that the average amplitude
            of the heterodyned data is independent of the SNR of the
            event.
        """
        coef = coef.copy()
        # Disable frequency cutoff
        ampcoef, phasecoef = like.waveform_model.split_amp_phase_coef(coef)
        ampcoef[-1] = np.inf
        coef = np.concatenate([ampcoef, phasecoef])
        h_df = like.waveform_model(
            like.event_data.frequencies[like.event_data.fslice], coef)

        amp_d = ampcoef[:like.waveform_model.n_det]

        # Downsample and rescale so that the amplitude is always similar.
        rb_splines = like.rb_splines.reinstantiate(
            fbin=None, pn_phase_tol=self.pn_phase_tol_compression)
        heterodyned_data = rb_splines.get_summary_weights(
            like.event_data.blued_strain[:, like.event_data.fslice]
            * h_df.conj()
            ) / amp_d[:, np.newaxis]**2 * 1e-4  # factor made up so ~ O(1)

        return heterodyned_data


def _check_sim_dir(sim_dir):
    new_filenames = ('simulation_data.npy',
                     'folded_sampled_params.npy',
                     'unfolding_labels.npy')
    existing = [path for filename in new_filenames
                if (path := sim_dir/filename).exists()]
    if existing:
        raise FileExistsError(f'{existing} already exist!')

    config_file = sim_dir/CONFIG_FILENAME
    if not config_file.exists():
        raise FileNotFoundError(f'Missing {config_file}')

    parameters_file = sim_dir/PARAMETERS_FILENAME
    if not parameters_file.exists():
        raise FileNotFoundError(
            f'Missing {parameters_file}, run `generate_parameters.py`.')


def submit_condor(sim_dir,
                  request_cpus,
                  request_memory='1G',
                  request_disk='1G',
                  **submit_kwargs):
    """
    Submit an HTCondor job to generate simulation parameters.

    This method generates 'simulation.{sub,sh,out,err,log}',
    files, the user should provide any instructions for the submit file
    as `**submit_kwargs`.

    Parameters
    ----------
    sim_dir: str, os.PathLike
        Simulations directory, should contain files `config.py` and
        `simulation_parameters.feather`.

    request_cpus, request_memory, request_disk: int or str
        Specifications in the HTCondor submit file.

    **submit_kwargs
        Further options to include in the HTCondor submit file. Do
        not pass `executable`, `output`, `error`, `log`, `args`,
        `queue`, which will be dealt with automatically.
    """
    sim_dir = Path(sim_dir).resolve()
    _check_sim_dir(sim_dir)

    submit_kwargs = {
        'submit_path': sim_dir/'simulation.sub',
        'executable': sim_dir/'simulation.sh',
        'output': sim_dir/'simulation.out',
        'error': sim_dir/'simulation.err',
        'log': sim_dir/'simulation.log',
        'args': f'{sim_dir} --processes {request_cpus}',
        'request_cpus': request_cpus,
        'request_memory': request_memory,
        'request_disk': request_disk,
        } | submit_kwargs

    cogwheel.utils.submit_condor(**submit_kwargs)


def main(sim_dir, processes=None):
    sim_dir = Path(sim_dir)
    _check_sim_dir(sim_dir)

    config = load_config(sim_dir/CONFIG_FILENAME)
    simulation_parameters = pd.read_feather(sim_dir/PARAMETERS_FILENAME)

    simulator = Simulator(config.EVENT_DATA_KWARGS, config.APPROXIMANT)

    dummy_event_data = data.EventData.gaussian_noise(
        **config.EVENT_DATA_KWARGS)
    waveform_model = PhenomenologicalWaveformGenerator.from_event_data(
        event_data=dummy_event_data, pn_phase_tol=0.1)

    data_preprocessor = DataPreprocessor(
        waveform_model,
        i_refdet=get_i_refdet(config),
        pn_phase_tol_compression=config.PN_PHASE_TOL_COMPRESSION)

    simulation_data, folded_sampled_params, unfolding_labels, derived \
        = simulate_and_preprocess_samples(
            simulator,
            data_preprocessor,
            simulation_parameters,
            transform_dic=get_transform_dic(config),
            processes=processes)

    np.save(sim_dir/'simulation_data.npy', simulation_data)
    np.save(sim_dir/'folded_sampled_params.npy', folded_sampled_params)
    np.save(sim_dir/'unfolding_labels.npy', unfolding_labels)

    cogwheel.utils.update_dataframe(simulation_parameters, derived)
    simulation_parameters.to_feather(sim_dir/PARAMETERS_FILENAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate signals to generate training data.')
    parser.add_argument(
        'sim_dir',
        help='''Simulation directory path, must contain files
                `config.py`. and `simulation_parameters.feather`.''')

    parser.add_argument('--processes', type=int, help='Number of processes')
    main(**vars(parser.parse_args()))
