"""
This module provides a semi-coherent likelihood function that can be
maximized to define a reference waveform.
It uses a phenomenological waveform model with few, uncorrelated
parameters.
"""
from scipy import interpolate, optimize
import numpy as np

import lal


def get_unwrapped_phase(frequencies, signal, mchirp):
    """
    Avoid unwrapping artifacts by dechirping with an approximate phase,
    unwrapping, and adding back the approximate phase.
    """
    phase_0 = -3/128 * (np.pi * mchirp * lal.MTSUN_SI * frequencies) ** (-5/3)
    dechirped = signal * np.exp(-1j*phase_0)
    return np.unwrap(np.angle(dechirped)) + phase_0


class SemicoherentLikelihood:
    """Methods to fit a phenomenological waveform to the data."""

    def __init__(self, event_data, ref_waveform_phase, waveform_model,
                 n_coherent_segments):
        """
        Parameters
        ----------
        event_data: cogwheel.data.EventData
            Contains data, i.e. signal plus noise.

        ref_waveform_phase: float array of shape (n_det, n_freq)
            Must be defined on ``event_data.frequencies[event_data.fslice]``.
            The function ``get_unwrapped_phase`` may be helpful for this.

        waveform_model: waveform_model.PhenomenologicalWaveformGenerator
            Will be used to generate a relative binning reference.

        n_coherent_segments: int
            The frequency range is partitioned into segments, a constant
            phase is optimized independently in each segment. This is
            unphysical and intended to make the maximization more robust
            to limitations in the phase model.
            ``n_coherent_segments=1`` corresponds to fully coherent.
        """
        assert (ref_waveform_phase.shape
                == event_data.strain[:, event_data.fslice].shape)

        self.event_data = event_data
        self.ref_waveform_phase = ref_waveform_phase
        self.waveform_model = waveform_model

        self._coherent_segment_inds = None  # Set by n_coherent_segments.setter
        self.n_coherent_segments = n_coherent_segments

        self._d_h_weights = None  # Set by `._set_summary()`
        self._h_h_weights = None  # Set by `._set_summary()`
        self._set_summary()

    @property
    def n_coherent_segments(self):
        """Number of coherent segments."""
        return self._n_coherent_segments

    @n_coherent_segments.setter
    def n_coherent_segments(self, n_coherent_segments):
        self._n_coherent_segments = n_coherent_segments
        self._coherent_segment_inds = np.array_split(
            np.arange(len(self.rb_splines.fbin)),
            self.n_coherent_segments)

    @property
    def rb_splines(self):
        """Splines for relative binning compression."""
        return self.waveform_model.phase_model.rb_splines

    @property
    def frequencies(self):
        """RFFT frequencies, with highpass slice applied."""
        return self.event_data.frequencies[self.event_data.fslice]

    @property
    def wht_filter(self):
        """
        Whitening filter of shape (n_det, n_freq), defined on
        ``.frequencies``.
        """
        return self.event_data.wht_filter[:, self.event_data.fslice]

    def semicoherent_lnlike(self, shapecoef):
        """
        Maximize over phase in each coherent segment & detector.
        Maximize over amplitude at each detector.
        """
        _, hh_d, dh_semicoherent_d = self._get_dh_hh(shapecoef)
        return np.sum(dh_semicoherent_d**2 / hh_d) / 2

    def fit_coef(self, frequencies, *, ref_waveform_phase,
                 ref_waveform_amp):
        """
        Find a phenomenological waveform that fits the data by starting
        from a guess that is matched to a user-provided waveform and
        refining the guess to maximize the semi-coherent likelihood.

        Parameters
        ----------
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
        coef: float array
            Parameters of the best-fit phenomenological waveform.

        dh_semicoherent: float array of shape (n_det,)
            Semicoherent ⟨d|h⟩ of the best fit waveform.

        h_h: float array of shape (n_det,)
            ⟨h|h⟩ of the best fit waveform.
        """
        assert np.array_equal(frequencies, self.frequencies)

        shapecoef_guess = self.waveform_model.guess_shapecoef(
            frequencies,
            self.wht_filter,
            ref_waveform_phase=ref_waveform_phase,
            ref_waveform_amp=ref_waveform_amp)

        shapeampcoef_bounds = self.waveform_model.amplitude_model \
            .amplitude_tapering.shapeampcoef_bounds

        shapecoef = optimize.minimize(
            lambda shapecoef: -self.semicoherent_lnlike(shapecoef),
            x0=shapecoef_guess,
            tol=.1,
            bounds=[*shapeampcoef_bounds,
                    *[(-np.inf, np.inf)] * (len(shapecoef_guess)
                                            - len(shapeampcoef_bounds))]
            ).x
        return self._fit_amp_phase(shapecoef)

    def _fit_amp_phase(self, shapecoef):
        """
        Find best fit amplitude and phase given a waveform shape.

        Parameters
        ----------
        shapecoef: float array
            Coefficients characterizing the waveform shape.
            You may use the output of ``._guess_shapecoef`` for this.

        Return
        ------
        coef: float array
            `shapecoef` but with additional entries for detector
            amplitudes and phases that maximize the likelihood.
            Can be passed to ``.waveform_model`` to produce a waveform.

        dh_semicoherent: float array of shape (n_det,)
            Semicoherent ⟨d|h⟩ of the best fit waveform.

        h_h: float array of shape (n_det,)
            ⟨h|h⟩ of the best fit waveform.
        """
        dh_d, hh_d, dh_semicoherent_d = self._get_dh_hh(shapecoef)

        best_phase = np.angle(dh_d)
        best_amp = np.abs(dh_semicoherent_d) / hh_d

        coef = self.waveform_model.coef_from_shapecoef(shapecoef,
                                                       det_amp=best_amp,
                                                       det_phase=best_phase)
        dh_semicoherent = np.abs(dh_semicoherent_d) * best_amp
        h_h = hh_d * best_amp**2
        return coef, dh_semicoherent, h_h

    def _get_dh_hh(self, shapecoef):
        """With fiducial amp_det=1, phase_det=0."""
        h_df = self.waveform_model.waveform_fiducial_amp_and_phase(
            self.rb_splines.fbin, shapecoef)
        dh_df = self._d_h_weights * h_df.conj()
        dh_d = np.sum(dh_df, axis=1)

        dh_semicoherent_d = np.sum([np.abs(np.sum(dh_df[:, inds], axis=1))
                                     for inds in self._coherent_segment_inds],
                                    axis=0)
        hh_d = np.sum(self._h_h_weights * (h_df.real**2 + h_df.imag**2),
                      axis=1)
        return dh_d, hh_d, dh_semicoherent_d

    def _set_summary(self):
        phase_fbin = interpolate.make_interp_spline(
            self.event_data.frequencies[self.event_data.fslice],
            self.ref_waveform_phase, k=1, axis=1
            )(self.rb_splines.fbin)

        # Assume amplitude will be smooth, set to 1 in reference waveform:
        h0_f = np.exp(1j * self.ref_waveform_phase)
        h0_fbin = np.exp(1j * phase_fbin)

        self._d_h_weights = self.rb_splines.get_summary_weights(
            self.event_data.blued_strain[:, self.event_data.fslice]
            * h0_f.conj()) / h0_fbin.conj()

        self._h_h_weights = self.rb_splines.get_summary_weights(
            self.wht_filter**2)

    def get_heterodyned_data_and_signal(self, coef, pn_phase_tol=None):
        """
        Parameters
        ----------
        coef: float array
            Parameters of the best-fit phenomenological waveform, that
            will be used to heterodyne the data.

        pn_phase_tol: float, optional
            Inversely proportional to the frequency resolution of the
            heterodyned data.

        Return
        ------
        heterodyned_data: complex array of shape (n_det, n_freq)
            Data, heterodyned with a reference waveform defined by
            `coef`. The frequency cutoff parameter is ignored in the
            reference waveform, to preserve high-frequency data.
            The amplitude is canceled out so that the average amplitude
            of the heterodyned data is independent of the SNR of the
            event.

        heterodyned_signal: complex array of shape (n_det, n_freq)
            Similar to `heterodyned_data` but with the noise realization
            subtracted. Note, this information is inaccesible except in
            simulations.

        fbin: float array of shape (n_freq,)
            Frequencies at which the heterodyned data are evaluated.
        """
        h_df = self.waveform_model(
            self.event_data.frequencies[self.event_data.fslice], coef,
            apply_tapering=False)

        amp_d = coef[:self.waveform_model.n_det]

        # Define coarse frequency grid:
        if pn_phase_tol is None:
            rb_splines = self.rb_splines
        else:
            rb_splines = self.rb_splines.reinstantiate(
                fbin=None, pn_phase_tol=pn_phase_tol)

        def heterodyne(event_data):
            # Downsample and rescale so amplitude is always similar:
            return rb_splines.get_summary_weights(
                event_data.blued_strain[:, event_data.fslice] * h_df.conj()
                ) / amp_d[:, np.newaxis]**2 * 1e-4  # factor made up so ~ O(1)

        heterodyned_data = heterodyne(self.event_data)

        event_data_noiseless = self.event_data.reinstantiate(
            strain=np.zeros_like(self.event_data.strain), injection=None)
        # This recomputes the waveform; if it ever becomes a bottleneck
        # we may want to restructure the code:
        event_data_noiseless.inject_signal(
            self.event_data.injection['par_dic'],
            self.event_data.injection['approximant'])
        heterodyned_signal = heterodyne(event_data_noiseless)

        return heterodyned_data, heterodyned_signal, rb_splines.fbin
