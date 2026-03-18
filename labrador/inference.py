"""
Setup inference for a particular event after SBI model has been trained.

This includes finding a reference waveform.
"""
import argparse
from pathlib import Path
import sys
from scipy import interpolate

from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import cogwheel.waveform
import cogwheel.data
import cogwheel.gw_plotting
import cogwheel.posterior

from labrador.reference import get_unwrapped_phase
from labrador import compression, simulation, utils, posterior, rbsplines


def get_compressed_data_and_transform(tree, event_data, frequencies,
                                      ref_amp, ref_phase,
                                      ret_phase=False):
    """
    Compress an event's data by SVD projection of the heterodyned data.

    This function optimizes the reference waveform starting from a
    guess, heterodynes the strain data against this reference, then
    projects the heterodyned data onto an SVD basis and concatenates the
    (suitably processed) parameters of the reference waveform.

    Parameters
    ----------
    tree : utils.Tree
        Contains directory tree of trained `labrador` model.

    event_data : cogwheel.data.EventData
        Data to be compressed.

    frequencies, ref_amp, ref_phase : array
        Guess for the reference amplitude and phase, to initialize the
        optimization. See `get_ref_amp_phase`.

    ret_phase : bool
        Whether to return the optimized phase (for testing purposes).

    Returns
    -------
    compressed_data : (1, n_features) torch.Tensor
        Input to the labrador posterior.

    transform : transform.TransformMixin
        Input to the labrador posterior.

    phase : (n_det, n_freq) array
        Frequency dependent phase of the optimized reference waveform
        (rad), defined on ``event_data.frequencies[event_data.fslice]``.
        Only returned if `ret_phase` is True.

        Hint: visualize the heterodyned data with

        .. code-block:: python

            event_data.heterodyne(phase).specgram((-0.1, 0.1))

    See Also
    --------
    posterior.Posterior.generate_samples_and_lnprob

    posterior.ImportancePosterior.get_weighted_samples_and_lnz
    """
    _, data_preprocessor, transform_class = simulation.setup_simulator(
        tree.rundir)

    preprocessed_data, transform_kwargs = _preprocess_data(
        data_preprocessor=data_preprocessor,
        event_data=event_data,
        frequencies=frequencies,
        ref_amp=ref_amp,
        ref_phase=ref_phase,
    )

    _print_hh_message(event_data.detector_names, preprocessed_data['h0_h0'])

    compressed_data = compression.compress_data(
        tree.rundir,
        preprocessed_data['heterodyned_data'],
        preprocessed_data['processed_coef'],
    )
    transform = transform_class(**transform_kwargs)

    if not ret_phase:
        return compressed_data, transform

    _, phasecoef = data_preprocessor.waveform_model.split_amp_phase_coef(
        preprocessed_data['coef'])
    phase = data_preprocessor.waveform_model.phase_model(frequencies,
                                                         phasecoef)
    return compressed_data, transform, phase


def _preprocess_data(data_preprocessor,
                     event_data,
                     frequencies,
                     ref_amp,
                     ref_phase,
                     n_trials=4, n_repeats=1, hh_tol=1):
    """
    Run repeated preprocessing trials and pick the best result.

    The function calls `data_preprocessor.preprocess_data` multiple
    times to mitigate stochastic variation in maximization.
    Trials are repeated until the best ⟨h|h⟩ is repeated `n_repeats`
    times to within a tolerance `hh_tol`.

    Parameters
    ----------
    data_preprocessor : simulation.DataPreprocessor

    event_data : cogwheel.data.EventData
        Contains detector strain and metadata.

    frequencies : (n_freq,) array
        Frequencies on which `ref_amp`, `ref_phase` are defined.
        Must match `event_data.frequencies[event_data.fslice]`.

    ref_amp, ref_phase : (n_det, n_freq) arrays
        Guess for the reference amplitude and phase, to initialize the
        optimization. See `get_ref_amp_phase`.

    n_trials : int
        Minimum number of preprocessing attempts per iteration.

    n_repeats : int
        Number of top results to compare for convergence.

    hh_tol : float
        Convergence tolerance on ⟨h|h⟩ differences.

    Returns
    -------
    preprocessed_data : dict
        The best preprocessed data product selected among trials. In
        particular, it contains 'heterodyned_data' and 'processed_coef'
        that are inputs to `compression.compress_data`.

    transform_kwargs : dict
        Keyword arguments to instantiate the transform.
    """
    results = []
    while True:
        results.extend([data_preprocessor.preprocess_data(event_data,
                                                          frequencies,
                                                          ref_amp,
                                                          ref_phase
                                                         )
                        for _ in range(n_trials)])
        h_h = [preproc_data['h0_h0'].sum() for (preproc_data, _) in results]
        hh_decreasing = np.sort(h_h)[::-1]
        if hh_decreasing[0] - hh_decreasing[n_repeats] < hh_tol:
            break
        n_trials = 1

    i_best = np.argmax(h_h)
    preprocessed_data, transform_kwargs = results[i_best]

    return preprocessed_data, transform_kwargs


def _print_hh_message(detector_names, h_h):
    terms = " + ".join(f"{v:.2f}" for v in h_h)
    print(f"Reference waveform ⟨h|h⟩ ({' + '.join(detector_names)} = total):\n"
          f"{terms} = {h_h.sum():.2f}")


def get_ref_amp_phase(data_config, event_data, m1, m2,
                      t_range=(-0.1, 0.1), **kwargs):
    """
    Return reference amplitude and phase to initialize maximization.

    Only the time coordinate is optimized, within t_range.

    Returns
    -------
    frequencies : (n_freq) float array
        Frequency grid on which ref_amp and ref_phase are defined (Hz).

    ref_amp : (n_det, n_freq) float array
        Reference amplitude A(f).

    ref_phase : (n_det, n_freq) float array
        Reference phase Φ(f), obtained from the intrinsic parameters
        passed and the optimized detector-arrival-times.

    See Also
    --------
    get_compressed_data_and_transform : Uses output of this function.
    """
    par_dic_guess = _build_par_dic(data_config, m1=m1, m2=m2, **kwargs)

    wfg = cogwheel.waveform.WaveformGenerator.from_event_data(
        event_data, approximant=data_config.APPROXIMANT)
    like = cogwheel.likelihood.CBCLikelihood(event_data, wfg)

    h_f = _generate_normalized_waveform(like, par_dic_guess)

    dt_det = _find_best_times(like, h_f, t_range)

    frequencies = event_data.frequencies[event_data.fslice]
    ref_phase = _compute_ref_phase(event_data, h_f, dt_det, par_dic_guess)
    ref_amp = np.broadcast_to(np.abs(h_f[:, event_data.fslice]),
                              ref_phase.shape)
    return frequencies, ref_amp, ref_phase


def _build_par_dic(data_config, **kwargs):
    """Return complete dictionary of waveform parameters."""
    default_pars = {
        'iota': 0.0,
        'f_ref': data_config.PRIOR_KWARGS['f_ref'],
        'phi_ref': 0.0,
        'd_luminosity': 1.0,
        **cogwheel.waveform.DEFAULT_PARS,
    }  # Unimportant for the reference waveform, but we need a value
    return default_pars | kwargs


def _generate_normalized_waveform(like, par_dic_guess):
    """Return normalized h_plus strain template."""
    event_data = like.event_data
    frequencies = event_data.frequencies[event_data.fslice]
    hplus, _ = like.waveform_generator.get_hplus_hcross(
        frequencies, par_dic_guess)
    h_f = np.zeros_like(event_data.strain)
    h_f[:, event_data.fslice] = hplus
    h_f /= np.sqrt(like._compute_h_h(h_f))[:, np.newaxis]
    return h_f


def _find_best_times(like, h_f, t_range, dt_tol=1e-3):
    """Return best time offset per detector from max SNR."""
    event_data = like.event_data
    i0 = np.searchsorted(event_data.times, event_data.tcoarse)
    dt = event_data.times[1] - event_data.times[0]
    i_range = i0 + (np.array(t_range) / dt).astype(int)
    # Assume the data can be cyclic (cogwheel injections are):
    t_inds = np.arange(*i_range) % len(event_data.times)

    t_values = (np.arange(*i_range) - i0) * dt

    z_cos, z_sin = np.array(like._matched_filter_timeseries(h_f))[:, :, t_inds]
    snr = np.abs(z_cos + 1j * z_sin)
    i_best_det, i_best_time = np.unravel_index(snr.argmax(), snr.shape)

    # Constrain other detectors to be within GW travel time of peak
    best_times = np.empty(len(event_data.detector_names))
    best_times[i_best_det] = t_values[i_best_time]
    best_det_name = event_data.detector_names[i_best_det]
    for i_det, det_name in enumerate(event_data.detector_names):
        if i_det == i_best_det:
            continue
        max_delay = dt_tol + cogwheel.gw_utils.detector_travel_times(
            det_name, best_det_name)
        mask = np.abs(t_values - t_values[i_best_time]) <= max_delay
        best_times[i_det] = t_values[mask][np.argmax(snr[i_det, mask])]

    return best_times


def _compute_ref_phase(event_data, h_f, dt_det, par_dic_guess):
    """Return reference phase array per detector and frequency."""
    mchirp_guess = cogwheel.gw_utils.m1m2_to_mchirp(
        par_dic_guess["m1"], par_dic_guess["m2"])
    freqs = event_data.frequencies[event_data.fslice]
    unwrapped_phase = get_unwrapped_phase(freqs,
                                          h_f[:, event_data.fslice],
                                          mchirp_guess)
    return unwrapped_phase - 2*np.pi*np.outer(dt_det, freqs)


def get_event_data_with_training_detectors(data_config, event):
    """
    Drop any extra detectors if the model was not trained with them.

    Parameters
    ----------
    data_config : module
        Output of :py:func:`utils.get_data_config`.

    event : cogwheel.data.EventData, str
        Event data or event name (the latter assumes the event is in
        `cogwheel.data.DATADIR`; if you install cogwheel from source you
        will get O1-O3 event data).

    Returns
    -------
    cogwheel.data.EventData : only contains the desired detectors.

    Raises
    ------
    ValueError : if there are no data for any of the desired detectors.
    """
    if isinstance(event, str):
        event_data = cogwheel.data.EventData.from_npz(eventname=event)
    elif isinstance(event, cogwheel.data.EventData):
        event_data = event
    else:
        raise ValueError('Expected `event` to be an EventData or str')

    detector_names = tuple(data_config.EVENT_DATA_KWARGS['detector_names'])
    if tuple(event_data.detector_names) == detector_names:
        return event_data

    try:
        det_inds = [event_data.detector_names.index(det)
                    for det in detector_names]
    except ValueError as err:
        raise ValueError(
            f'{event_data.eventname} does not have {detector_names} detectors'
        ) from err

    print(f'{event_data.eventname}: '
          f'downselecting from {event_data.detector_names} to '
          f'{detector_names} detectors with which model was trained.')
    return event_data.reinstantiate(detector_names=detector_names,
                                    strain=event_data.strain[det_inds],
                                    wht_filter=event_data.wht_filter[det_inds])


def mimic_fiducial_wht_filter(data_config, event_data):
    """
    Mimic the fiducial whitening filter at low resolution.

    Apply a smooth modification to the whitening filter of `event_data`
    to make it resemble the fiducial whitening filter (with which
    training is done) at low resolution, while preserving the sharp
    features of the actual one to prevent artifacts.

    This is done as a first patch to include PSD information without
    actually training on different PSDs. The implementation may change
    in the future.

    Parameters
    ----------
    data_config : module
        Output of utils.load_data_config

    event_data : cogwheel.data.EventData
        Contains a gravitational wave event (and a whitening filter).

    Returns
    -------
    cogwheel.data.EventData : With the modified whitening filter.
    """
    fiducial_event_data = cogwheel.data.EventData.gaussian_noise(
        **data_config.EVENT_DATA_KWARGS)

    rb_splines_fiducial = rbsplines.RelativeBinningSplines(
        fiducial_event_data.frequencies[fiducial_event_data.fslice],
        pn_phase_tol=1.
    )

    rb_splines = rbsplines.RelativeBinningSplines(
        event_data.frequencies[event_data.fslice], pn_phase_tol=1.
    )

    weight_fid = rb_splines_fiducial.get_summary_weights(
        fiducial_event_data.frequencies[fiducial_event_data.fslice] ** (-7/3)
        * fiducial_event_data.wht_filter[:, fiducial_event_data.fslice]**2
    )
    weight = rb_splines.get_summary_weights(
        event_data.frequencies[event_data.fslice] ** (-7/3)
        * event_data.wht_filter[:, event_data.fslice]**2
    )

    effective_wht_filter = event_data.wht_filter.copy()
    effective_wht_filter[:, event_data.fslice] *= np.sqrt(np.abs(
        interpolate.make_interp_spline(
            rb_splines.fbin, weight_fid / weight, axis=1
        )(event_data.frequencies[event_data.fslice])
    ))
    return event_data.reinstantiate(wht_filter=effective_wht_filter)


def _filter_samples_in_training_range(data_config, samples):
    """Discard samples outside training range for mchirp or q."""
    mchirp = cogwheel.gw_utils.m1m2_to_mchirp(**samples[['m1', 'm2']])
    q = samples['m2'] / samples['m1']
    mchirp_min, mchirp_max = data_config.PRIOR_KWARGS['mchirp_range']
    return samples[
        (mchirp > mchirp_min)
        & (mchirp < mchirp_max)
        & (q > data_config.PRIOR_KWARGS['q_min'])
    ]


def run_importance_sampling(tree, eventsdir, eventdata_path,
                            mchirp_guess=None):
    """
    Run labrador on a real event, including importance sampling.

    Parameters
    ----------
    tree : utils.Tree
        Specifies the labrador model to use.

    eventsdir : os.PathLike
        Output directory, will be created if it does not already exist.

    eventdata_path : os.PathLike
        Path to a cogwheel.data.EventData npz file.
    """
    event_data = cogwheel.data.EventData.from_npz(filename=eventdata_path)
    eventname = event_data.eventname
    eventdir = Path(eventsdir)/eventname
    eventdir.mkdir(parents=True)

    with redirect_output(eventdir/'log.txt'):
        post = posterior.Posterior.from_tree(tree)
        data_config = utils.load_data_config(tree.rundir)
        prior_config = utils.load_prior_config(tree.priordir)

        event_data = get_event_data_with_training_detectors(
            data_config, event_data)

        event_data_fid_wht_filter = mimic_fiducial_wht_filter(
            data_config, event_data)

        if mchirp_guess is None:
            mchirp_guess = cogwheel.data.EVENTS_METADATA.loc[
                eventname, 'mchirp']

        # Optimize time at each detector
        m_i = mchirp_guess * 2**.2
        kw = {'m1': m_i, 'm2': m_i}  # Basic guess, could make more detailed

        frequencies, ref_amp, ref_phase = get_ref_amp_phase(
            data_config, event_data, **kw, t_range=(-0.1, 0.1))

        compressed_data, transform, optimized_phase \
            = get_compressed_data_and_transform(
                tree, event_data_fid_wht_filter, frequencies, ref_amp,
                ref_phase, ret_phase=True)

        with PdfPages(eventdir/f'{eventname}.pdf') as pdf:
            # Spectrograms of heterodyned data
            ## 1. with the reference waveform we provide
            event_data.heterodyne(ref_phase).specgram((-.1, .1), nfft=16)
            pdf.savefig(bbox_inches='tight')
            ## 2. with fully-optimized reference phase:
            event_data.heterodyne(optimized_phase).specgram((-.1, .1), nfft=16)
            pdf.savefig(bbox_inches='tight')

            # Make cogwheel posterior
            cogwheel_posterior = cogwheel.posterior.Posterior.from_event(
                event_data,
                mchirp_guess,
                data_config.APPROXIMANT,
                prior_config.PRIOR_CLASS,
                ref_wf_finder_kwargs={
                    'f_ref': data_config.PRIOR_KWARGS['f_ref']}
            )

            # Importance sampling
            post_is = posterior.ImportancePosterior(
                post, compressed_data, transform, cogwheel_posterior)
            samples, _ = post_is.get_weighted_samples_and_lnz()
            samples.to_feather(eventdir/'samples.feather')
            # Discard samples outside training range, otherwise it's unfair
            samples = _filter_samples_in_training_range(data_config, samples)

            # Corner plots
            eff = cogwheel.utils.n_effective(samples['weights']) / len(samples)
            dataframes = {
                'labrador': samples.drop(columns='weights'),
                rf'labrador reweighted ($\epsilon = {eff:.3g}$)': samples,
            }

            standard_params = [
                par for par in transform.standard_params
                if not np.allclose(samples[par].iloc[0], samples[par])
            ]

            rescaled_params = [par for par in samples
                               if par.startswith('rescaled_')]

            cogwheel.gw_plotting.MultiCornerPlot(
                dataframes, params=standard_params, tail_probability=1e-3
            ).plot(title=f'{eventname} - Standard parameters')
            pdf.savefig(bbox_inches='tight')

            cogwheel.gw_plotting.MultiCornerPlot(
                dataframes, params=transform.sampled_params,
                tail_probability=1e-3
            ).plot(title=f'{eventname} - Transformed parameters')
            pdf.savefig(bbox_inches='tight')

            cogwheel.gw_plotting.MultiCornerPlot(
                dataframes, params=rescaled_params, tail_probability=1e-3
            ).plot(title=f'{eventname} - Rescaled parameters')
            pdf.savefig(bbox_inches='tight')


class redirect_output:
    """Redirect all stdout and stderr to the given log file."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_file = None
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def __enter__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        sys.stdout = self.log_file
        sys.stderr = self.log_file
        return self.log_file  # optional, e.g. for writing manually

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        if self.log_file is not None:
            self.log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run labrador and importance-sample against cogwheel.'
    )
    parser.add_argument('sbidir', help='Path to SBI directory')
    parser.add_argument('unfolderdir', help='Path to unfolder directory')
    parser.add_argument('eventsdir', help='Output directory')
    parser.add_argument('eventdata_path', type=Path,
                        help='Path to a cogwheel.data.EventData npz file.')
    parser.add_argument('--mchirp-guess', type=float, help='Chirp mass (M⊙)')

    args = parser.parse_args()
    dir_tree = utils.Tree(args.sbidir, args.unfolderdir)
    run_importance_sampling(dir_tree, args.eventsdir, args.eventdata_path,
                            args.mchirp_guess)
