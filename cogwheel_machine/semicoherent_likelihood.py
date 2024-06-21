"""
This module provides a semi-coherent likelihood function that can be
maximized to define a reference waveform.
It uses a phenomenological waveform model with few, uncorrelated
parameters.
"""
from scipy import interpolate, optimize
import numpy as np

import lal

from .rbsplines import RelativeBinningSplines


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
    @classmethod
    def from_event_data(cls,
                        event_data,
                        ref_waveform_phase,
                        waveform_model,
                        n_coherent_segments=8):
        """
        Parameters
        ----------
        event_data: cogwheel.data.EventData

        ref_waveform_phase: float array of shape (n_det, n_freq)
            Must be defined on ``event_data.frequencies[event_data.fslice]``.
            The function ``get_unwrapped_phase`` may be helpful for this.

        pn_phase_tol: float

        n_coherent_segments: int
            The frequency range is partitioned into segments, a constant
            phase is optimized independently in each segment. This is
            unphysical and intended to make the maximization more robust
            to limitations in the phase model.
            ``n_coherent_segments=1`` corresponds to fully coherent.
        """
        assert (ref_waveform_phase.shape
                == event_data.strain[:, event_data.fslice].shape)

        rb_splines = RelativeBinningSplines(
            event_data.frequencies[event_data.fslice],
            fbin=waveform_model.phase_model.fbin)
        return cls(event_data=event_data,
                   ref_waveform_phase=ref_waveform_phase,
                   waveform_model=waveform_model,
                   n_coherent_segments=n_coherent_segments,
                   rb_splines=rb_splines,)

    def __init__(self, event_data, ref_waveform_phase,
                 waveform_model, n_coherent_segments,
                 rb_splines):
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

        rb_splines: rbsplines.RelativeBinningSplines
            Determines the frequency resolution at which (d|h) and (h|h)
            are computed.
        """
        np.testing.assert_allclose(rb_splines.fbin,
                                   waveform_model.phase_model.fbin)
        assert (ref_waveform_phase.shape
                == event_data.strain[:, event_data.fslice].shape)

        self.event_data = event_data
        self.ref_waveform_phase = ref_waveform_phase
        self.waveform_model = waveform_model
        self.rb_splines = rb_splines

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
        """
        shapecoef_guess = self._guess_shapecoef(
            frequencies,
            ref_waveform_phase=ref_waveform_phase,
            ref_waveform_amp=ref_waveform_amp)

        shapecoef = optimize.minimize(
            lambda shapecoef: -self.semicoherent_lnlike(shapecoef),
            x0=shapecoef_guess,
            tol=.1,
            bounds=[(1., 3.5),
                    *[(-np.inf, np.inf)] * (len(shapecoef_guess) - 1)]
            ).x
        coef = self._fit_amp_phase(shapecoef)
        return coef

    def _guess_shapecoef(self, frequencies, *, ref_waveform_phase,
                        ref_waveform_amp):
        """
        Find amplitude and phase coefficients that best match a given
        waveform amplitude and phase.

        Return
        ------
        shapecoef: float array
        """
        # TODO generalize this to arbitrary frequencies
        assert np.array_equal(frequencies,
                              self.event_data.frequencies[self.event_data.fslice])

        ref_wf_phase_fbin = interpolate.make_interp_spline(
            frequencies, ref_waveform_phase, axis=1, k=1)(self.rb_splines.fbin)

        phasecoef_guess = self.waveform_model.phase_model.guess_phasecoef(
            ref_wf_phase_fbin)

        log10_fcut_guess = self.waveform_model.amplitude_model.guess_log10_fcut(
            frequencies,
            self.event_data.wht_filter[:, self.event_data.fslice],
            ref_waveform_amp)

        shapecoef_guess = np.concatenate(
            [[log10_fcut_guess],
             phasecoef_guess[self.waveform_model.phase_model.n_det:]])
        return shapecoef_guess

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
        """
        dh_d, hh_d, dh_semicoherent_d = self._get_dh_hh(shapecoef)

        best_phase = np.angle(dh_d)
        best_amp = np.abs(dh_semicoherent_d) / hh_d

        coef = self.waveform_model.coef_from_shapecoef(shapecoef,
                                                       det_amp=best_amp,
                                                       det_phase=best_phase)
        return coef

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
            self.event_data.blued_strain[:, self.event_data.fslice] * h0_f.conj()
            ) / h0_fbin.conj()

        self._h_h_weights = self.rb_splines.get_summary_weights(
            self.event_data.wht_filter[:, self.event_data.fslice]**2)
