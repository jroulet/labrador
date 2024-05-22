"""
Classes
-------
Simulator:
    Generate data similar to user input.

DataPreprocessor:
    Compress data by heterodyning against a reference waveform.
"""
import scipy.optimize
import numpy as np

from cogwheel import data
from cogwheel import gw_utils
from cogwheel import waveform

from . import semicoherent_likelihood


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

        dummy_event_data = data.EventData.gaussian_noise(**self.event_data_kwargs)
        self._waveform_generator = waveform.WaveformGenerator.from_event_data(
            dummy_event_data, approximant)

    def generate_data_and_reference_waveform(self, parameters):
        """
        Generate data similar to what a user would provide.

        Parameters
        ----------
        parameters: dict-like
            Physical parameters of the signal to simulate. Must contain keys
            for all ``._waveform_generator.params``.

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
                 n_coherent_segments=8,
                 pn_phase_tol_compression=1.0):
        """
        Parameters
        ----------
        waveform_model: waveform_model.PhenomenologicalWaveformGenerator
            Used to generate the reference waveform.

        n_coherent_segments: int
            When maximizing the likelihood to find a reference waveform,
            the frequency range is partitioned into segments and a constant
            phase is optimized independently in each segment. This is
            unphysical and intended to make the maximization more robust
            to limitations in the phase model.
            ``n_coherent_segments=1`` corresponds to fully coherent.

        pn_phase_tol_compression: float
            Controls the relative-binning frequency resolution used for
            compressing the data after the reference waveform has been found.
            Lower tolerance means higher resolution.
        """
        self.waveform_model = waveform_model
        self.n_coherent_segments = n_coherent_segments
        self.pn_phase_tol_compression = pn_phase_tol_compression

    def preprocess_data(self,
                        event_data,
                        frequencies,
                        ref_waveform_amp,
                        ref_waveform_phase):
        """
        Compress the data by heterodyning it against a phenomenological
        reference waveform.

        The phenomenological reference waveform is found by first fitting a
        reference provided by the user, and then optimizing a semi-coherent
        likelihood using that as initial guess.
        The purpose of this optimization is to be insensitive to how the user
        found their reference waveform: we cannot control this and so we
        cannot trust that the training will capture it.

        Parameters
        ----------
        event_data: cogwheel.data.EventData
            Data containing the event.

        frequencies: float array of shape (n_freq,)
            Frequency array on which the user's reference waveform is defined.
            For now, it must match ``event_data.frequencies[event_data.fslice]``.

        ref_waveform_amp: float array of shape (n_det, n_freq)
            User-provided reference waveform amplitude.

        ref_waveform_phase: float array of shape (n_det, n_freq)
            User-provided reference waveform unwrapped phase.

        Return
        ------
        preprocessed_data: float array
            Contains the real and imaginary part of the heterodyned data at low
            frequency resolution, the parameters of the phenomenological
            reference waveform, and a few extra features that summarize the
            detector amplitude, phase and time differences. 
        """
        # TODO generalize frequencies
        assert np.array_equal(frequencies,
                              event_data.frequencies[event_data.fslice])

        like = semicoherent_likelihood.SemicoherentLikelihood.from_event_data(
            event_data=event_data,
            ref_waveform_phase=ref_waveform_phase,
            waveform_model=self.waveform_model,
            n_coherent_segments=self.n_coherent_segments)

        shapecoef_guess = like.guess_shapecoef(frequencies,
                                               ref_waveform_phase,
                                               ref_waveform_amp)

        shapecoef = scipy.optimize.minimize(
            lambda shapecoef: -like.semicoherent_lnlike(shapecoef),
            x0=shapecoef_guess,
            tol=.1,
            bounds=[(1., 3.5),
                    *[(-np.inf, np.inf)]*(len(shapecoef_guess)-1)]
        ).x
        coef = like.fit_amp_phase(shapecoef)
        h_df = like.waveform_model(
            like.event_data.frequencies[like.event_data.fslice], coef)

        # Downsample
        rb_splines = like.rb_splines.reinstantiate(
            fbin=None, pn_phase_tol=self.pn_phase_tol_compression)
        heterodyned_data = rb_splines.get_summary_weights(
            like.event_data.blued_strain[:, like.event_data.fslice] * h_df.conj())
        geometry_features = like.waveform_model.get_geometry_features(coef)
        preprocessed_data = np.concatenate([heterodyned_data.real.flat,
                                            heterodyned_data.imag.flat,
                                            coef,
                                            geometry_features])
        return preprocessed_data
