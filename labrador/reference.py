"""
Optimize the likelihood to find a reference waveform.

Use a phenomenological waveform model with few, uncorrelated parameters.
"""
from scipy import interpolate, optimize
from scipy.stats import qmc
import numpy as np

from cogwheel.utils import import_lal
lal = import_lal()


def get_unwrapped_phase(frequencies, signal, mchirp):
    """
    Avoid unwrapping artifacts by dechirping with an approximate phase,
    unwrapping, and adding back the approximate phase.
    """
    phase_0 = -3/128 * (np.pi * mchirp * lal.MTSUN_SI * frequencies) ** (-5/3)
    dechirped = signal * np.exp(-1j*phase_0)
    return np.unwrap(np.angle(dechirped)) + phase_0


def _get_differential_evolution_initial_population(bounds, popsize=15):
    n_dim = len(bounds)
    return qmc.scale(qmc.Halton(n_dim).random(popsize * n_dim), *zip(*bounds))


class ReferenceWaveformFinder:
    """Methods to fit a phenomenological waveform to the data."""

    def __init__(self, event_data, waveform_model):
        """
        Parameters
        ----------
        event_data : cogwheel.data.EventData
            Contains data, i.e. signal plus noise.

        waveform_model : waveform_model.PhenomenologicalWaveformGenerator
            Will be used to generate a relative binning reference.
        """
        self.event_data = event_data
        self.waveform_model = waveform_model

        self._d_h_weights = None  # Set by `._set_summary()`
        self._h_h_weights = None  # Set by `._set_summary()`

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
        (n_det, n_freq) whitening filter defined on ``.frequencies``.
        """
        return self.event_data.wht_filter[:, self.event_data.fslice]

    def fit_coef(self, frequencies, ref_waveform_phase):
        """
        Find a phenomenological waveform that fits the data by starting
        from a guess that is matched to a user-provided waveform and
        refining the guess to maximize the likelihood.

        Parameters
        ----------
        frequencies : float array of shape (n_freq,)
            Frequency array on which the user's reference waveform is
            defined. For now, it must match
            ``event_data.frequencies[event_data.fslice]``.

        ref_waveform_phase : float array of shape (n_det, n_freq)
            User-provided reference waveform unwrapped phase. The time
            convention is relative to ``.event_data.tgps``.

        Returns
        -------
        coef : float array
            Parameters of the best-fit phenomenological waveform.

        h_h : float array of shape (n_det,)
            ⟨h|h⟩ of the best fit waveform.

        f_cut : float
            Best-fit cutoff frequency (Hz).
        """
        assert np.array_equal(frequencies, self.frequencies)
        self._set_summary(ref_waveform_phase)

        big_boxsize = 20.0  # Hard bounds for optimization
        small_boxsize = 2.5  # Initial bounds for population

        shapecoef_guess = self.waveform_model.guess_shapecoef(
            frequencies, ref_waveform_phase)

        def get_bounds(boxsize):
            return shapecoef_guess[:, np.newaxis] + (-boxsize, boxsize)

        small_bounds = get_bounds(small_boxsize)
        big_bounds = get_bounds(big_boxsize)

        init_pop = _get_differential_evolution_initial_population(small_bounds)

        shapecoef = optimize.differential_evolution(
            lambda shapecoef: -self._get_snrsq_dh_hh_fcut(shapecoef)[0],
            bounds=big_bounds,
            init=init_pop,
            x0=shapecoef_guess,
        ).x

        return self._fit_amp_phase_fcut(shapecoef)

    def _fit_amp_phase_fcut(self, shapecoef):
        """
        Find best fit amplitude, phase and f_cut given a waveform shape.

        Parameters
        ----------
        shapecoef : float array
            Coefficients characterizing the waveform shape.
            You may use the output of ``._guess_shapecoef`` for this.

        Returns
        -------
        coef : float array
            `shapecoef` but with additional entries for detector
            amplitudes and phases that maximize the likelihood.
            Can be passed to ``.waveform_model`` to produce a waveform.

        h_h : float array of shape (n_det,)
            ⟨h|h⟩ of the best fit waveform.
        """
        _, dh_d, hh_d, f_cut = self._get_snrsq_dh_hh_fcut(shapecoef)

        best_phase = np.angle(dh_d)
        best_amp = np.abs(dh_d) / hh_d

        coef = self.waveform_model.coef_from_shapecoef(shapecoef,
                                                       det_amp=best_amp,
                                                       det_phase=best_phase)
        best_hh_d = hh_d * best_amp**2
        return coef, best_hh_d, f_cut

    def _get_snrsq_dh_hh_fcut(self, shapecoef):
        """
        Optimize likelihood over amplitude, phase and cutoff frequency.

        Returns
        -------
        snrsq : float
            Signal-to-noise ratio squared (= 2 ln L).

        d_h : complex
            (d|h) of the waveform at fiducial amp=1, phase=0.

        h_h : float
            ⟨h|h⟩ of the waveform at fiducial amp=1.

        f_cut : float
            Best-fit cutoff frequency (Hz).
        """
        h_df = self.waveform_model.waveform_fiducial_amp_and_phase(
            self.rb_splines.fbin, shapecoef)

        # Vary cutoff frequency
        cumdh_df = np.cumsum(self._d_h_weights * h_df.conj(), axis=1)
        cumhh_df = np.cumsum(self._h_h_weights * (h_df.real**2 + h_df.imag**2),
                      axis=1)
        snrsq_f = np.sum(
            (cumdh_df.real**2 + cumdh_df.imag**2) / cumhh_df, axis=0)
        i_cut = np.argmax(snrsq_f)
        f_cut = self.rb_splines.fbin[i_cut]

        return snrsq_f[i_cut], cumdh_df[:, i_cut], cumhh_df[:, i_cut], f_cut

    def _set_summary(self, ref_waveform_phase):
        assert (ref_waveform_phase.shape
                == self.event_data.strain[:, self.event_data.fslice].shape)

        # Use convention that waveform time is relative to event_data.tgps,
        # i.e., event_data.tcoarse in the strain segment.
        centered_blued_strain = (
            self.event_data.blued_strain[:, self.event_data.fslice]
            * np.exp(2j*np.pi * self.frequencies * self.event_data.tcoarse)
        )

        phase_fbin = interpolate.make_interp_spline(
            self.frequencies, ref_waveform_phase, k=1, axis=1
        )(self.rb_splines.fbin)

        # Assume amplitude will be smooth, set to 1 in reference waveform:
        h0_f = np.exp(1j * ref_waveform_phase)
        h0_fbin = np.exp(1j * phase_fbin)

        self._d_h_weights = self.rb_splines.get_summary_weights(
            centered_blued_strain * h0_f.conj()
        ) / h0_fbin.conj()

        self._h_h_weights = self.rb_splines.get_summary_weights(
            self.wht_filter**2)

    def get_heterodyned_data_and_signal(self, coef, pn_phase_tol=None):
        """
        Parameters
        ----------
        coef : float array
            Parameters of the best-fit phenomenological waveform, that
            will be used to heterodyne the data.

        pn_phase_tol : float, optional
            Inversely proportional to the frequency resolution of the
            heterodyned data.

        Returns
        -------
        heterodyned_data : complex array of shape (n_det, n_freq)
            Data, heterodyned with a reference waveform defined by
            `coef`. The frequency cutoff parameter is ignored in the
            reference waveform, to preserve high-frequency data.
            The amplitude is canceled out so that the average amplitude
            of the heterodyned data is independent of the SNR of the
            event.

        heterodyned_signal : complex array of shape (n_det, n_freq)
            Similar to `heterodyned_data` but with the noise realization
            subtracted. Note, this information is inaccesible except in
            simulations.

        fbin : float array of shape (n_freq,)
            Frequencies at which the heterodyned data are evaluated.
        """
        h_df = self.waveform_model(
            self.event_data.frequencies[self.event_data.fslice], coef)

        amp_d = coef[:self.waveform_model.n_det]

        # Define coarse frequency grid:
        if pn_phase_tol is None:
            rb_splines = self.rb_splines
        else:
            rb_splines = self.rb_splines.reinstantiate(
                fbin=None, pn_phase_tol=pn_phase_tol)

        shift = np.exp(2j*np.pi * self.frequencies * self.event_data.tcoarse)

        def heterodyne(event_data):
            # Downsample and rescale so amplitude is always similar:
            return rb_splines.get_summary_weights(
                shift * event_data.blued_strain[:, event_data.fslice]
                * h_df.conj()
            ) / amp_d[:, np.newaxis]**2 * 1e-4  # factor made up so ~ O(1)

        heterodyned_data = heterodyne(self.event_data)

        if self.event_data.injection:
            event_data_noiseless = self.event_data.reinstantiate(
                strain=np.zeros_like(self.event_data.strain), injection=None)
            # This recomputes the waveform; if it ever becomes a bottleneck
            # we may want to restructure the code:
            event_data_noiseless.inject_signal(
                self.event_data.injection['par_dic'],
                self.event_data.injection['approximant'])
            heterodyned_signal = heterodyne(event_data_noiseless)
        else:
            heterodyned_signal = None

        return heterodyned_data, heterodyned_signal, rb_splines.fbin
