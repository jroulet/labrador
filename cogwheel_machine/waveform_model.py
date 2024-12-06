"""
Phenomenological waveform model that works with coordinates
that are approximately orthonormal (under a reference PSD).
"""
from pathlib import Path
import scipy.interpolate
import scipy.optimize
from scipy.stats import qmc
import numpy as np
import pandas as pd

import lal

import cogwheel.data
import cogwheel.gw_utils
import cogwheel.waveform

from .rbsplines import RelativeBinningSplines
from . import utils


class PhenomenologicalWaveformGenerator:
    """
    Class that implementes a simple waveform model with the purpose of
    finding a reference waveform quickly.

    The phenomenological waveform is based on the 1.5pN expression for
    the phase and the 0pN with a phenomenological cutoff for the
    amplitude. The parameters of the waveform are named `coef` in the
    code. `coef` are a concatenation of `ampcoef` and `phasecoef`.
    `ampcoef` is an array of ``n_detectors + 1`` elements: the overall
    amplitude at each detector plus the cutoff frequency.
    `phasecoef` is an array of ``2*n_detectors + 2`` elements: the
    overall phase at each detector, the overall time (orthogonalized to
    phase) at each detector, and two coefficients that encode
    information about the intrinsic parameters (orthogonalized to time,
    phase, and each other). Also `shapecoef` are defined, which are the
    same as `coef` but excluding the amplitude and phase parameters,
    because these can be maximized over analytically.
    The likelihood should be rather uncorrelated in these parameters,
    easing the task of maximizing it.
    """
    @classmethod
    def from_rundir(cls, rundir, n_svd_examples=1000):
        """
        Parameters
        ----------
        rundir: os.PathLike
            Path to a directory on which ``generate_parameters`` has
            been run.

        n_svd_examples: int
            How many waveforms to simulate to input in the SVD of
            amplitude profiles.
        """
        rundir = Path(rundir)
        config = utils.load_data_config(rundir)
        dummy_event_data = cogwheel.data.EventData.gaussian_noise(
            **config.EVENT_DATA_KWARGS)

        frequencies = dummy_event_data.frequencies[dummy_event_data.fslice]
        wht_filter = dummy_event_data.wht_filter[:, dummy_event_data.fslice]

        waveform_generator \
            = cogwheel.waveform.WaveformGenerator.from_event_data(
                dummy_event_data, config.APPROXIMANT)

        simulation_parameters = pd.read_feather(
            rundir/utils.TRAINING_DIR/utils.PARAMETERS_FILENAME
            )[:n_svd_examples]

        return cls.from_waveforms(frequencies, wht_filter, waveform_generator,
                                  simulation_parameters, config.PN_PHASE_TOL)

    @classmethod
    def from_waveforms(cls, frequencies, fiducial_wht_filter,
                       waveform_generator, simulation_parameters,
                       pn_phase_tol):
        """
        Parameters
        ----------
        frequencies: (n_freq,) float array

        fiducial_wht_filter: (n_det, n_freq) float array

        waveform_generator: cogwheel.waveform.WaveformGenerator

        simulation_parameters: pandas.DataFrame
            Contains parameters of waveforms to simulate, to input in
            the SVD of amplitude profiles.

        pn_phase_tol: float
            Determines the internal frequency resolution at which the
            phase model will compute inner products in order to
            orthogonalize the phase bases. Lower tolerance means higher
            resolution.

        See Also
        --------
        .from_rundir
        """
        amplitude_tapering = AmplitudeTapering.from_scratch(
            waveform_generator, simulation_parameters)
        amplitude_model = AmplitudeModel(len(fiducial_wht_filter),
                                         amplitude_tapering)
        phase_model = PhaseModel.from_scratch(
            frequencies=frequencies,
            fiducial_wht_filter=fiducial_wht_filter,
            pn_phase_tol=pn_phase_tol)
        return cls(amplitude_model, phase_model)

    def __init__(self, amplitude_model, phase_model):
        """
        Parameters
        ----------
        amplitude_model: AmplitudeModel
            Phenomenological model for the amplitude of the waveform.

        phase_model: PhaseModel
            Phenomenological model for the phase of the waveform.
        """
        assert phase_model.n_det == amplitude_model.n_det
        self.amplitude_model = amplitude_model
        self.phase_model = phase_model

    def __call__(self, frequencies, coef, apply_tapering=True):
        """
        Parameters
        ----------
        frequencies: float array of shape (n_frequencies,)
            Evaluation frequencies (Hz).

        coef: float array of shape (`.n_coef`,)
            ampcoef, phasecoef concatenated.

        Return
        ------
        complex array of shape (n_det, n_frequencies)
            Waveform at detectors.
        """
        assert coef.shape == (self.n_coef,)
        ampcoef, phasecoef = self.split_amp_phase_coef(coef)
        amplitude = self.amplitude_model(frequencies, ampcoef, apply_tapering)
        phase = self.phase_model(frequencies, phasecoef)
        return amplitude * np.exp(1j*phase)

    def split_amp_phase_coef(self, coef):
        """Return `ampcoef`, `phasecoef` from `coef`."""
        return np.split(coef, [self.amplitude_model.n_ampcoef])

    def waveform_fiducial_amp_and_phase(self, frequencies, shapecoef):
        """
        Similar to `.__call__` but it takes fewer parameters, and sets
        detector amplitudes to 1 and detector phases to 0.

        (Useful because these can be maximized analytically.)

        Parameters
        ----------
        frequencies: float array of shape (n_frequencies,)
            Evaluation frequencies (Hz).

        shapecoef: float array of shape (`.n_coef` - 2*n_det,)
            ampcoef, phasecoef concatenated, but with entries
            corresponding to detector amplitude and phase removed.

        Return
        ------
        complex array of shape (n_det, n_frequencies)
            Waveform at detectors.
        """
        coef = self.coef_from_shapecoef(shapecoef,
                                        det_amp=np.ones(self.n_det),
                                        det_phase=np.zeros(self.n_det))
        return self(frequencies, coef)

    def coef_from_shapecoef(self, shapecoef, det_amp, det_phase):
        """
        Insert overall amplitude at each detector and phase at each
        detector into the array of shape coefficients, to generate a
        complete array of coefficients.

        Parameters
        ----------
        shapecoef: float array
            Waveform parameters other than detectors' amplitude and
            phase.

        det_amp: float array of shape (n_det,)
            Overall amplitude at each detector.

        det_phase: float array of shape (n_det,)
            Overall phase at each detector.

        Return
        ------
        coef: float array
        """
        assert shapecoef.shape == (self.n_coef - 2*self.n_det,)

        n_shapeampcoef = self.amplitude_model.n_ampcoef - self.n_det
        shapeampcoef, shapephasecoef = np.split(shapecoef, [n_shapeampcoef])
        ampcoef = np.concatenate([det_amp, shapeampcoef])
        phasecoef = np.concatenate([self.phase_model.detphasecoef(det_phase),
                                    shapephasecoef])
        coef = np.concatenate([ampcoef, phasecoef])
        return coef

    @property
    def n_coef(self):
        """Number of phenomenological parameters, ``coef``."""
        return self.amplitude_model.n_ampcoef + self.phase_model.n_phasecoef

    @property
    def n_det(self):
        """Number of detectors."""
        return self.amplitude_model.n_det

    def process_coef(self, coef, i_refdet):
        """
        Return array with the same information as `coef` but transformed
        in a way that makes it more suitable for a neural network.

        Parameters
        ----------
        coef: float array of shape (`.n_coef`,)
            ampcoef, phasecoef concatenated.

        Return
        ------
        float32 array
            A concatenation of the following quantities:
            * amp_rms                                         1
            * amp_ratios                                  n_det
            * cos(phase_differences)      n_det * (n_det-1) / 2
            * sin(phase_differences)      n_det * (n_det-1) / 2
            * time_differences            n_det * (n_det-1) / 2
            * intrinsic                                       3
            * cos(ref_det_phase)                              1
            * sin(ref_det_phase)                              1
            * ref_det_time                                    1
        """
        ampcoef, phasecoef = np.split(coef, [self.amplitude_model.n_ampcoef])
        amp_rms, amp_ratios \
            = self.amplitude_model.get_detector_amp_rms_and_ratios(ampcoef)

        det_phase, det_time = self.phase_model.get_detector_phases_and_times(
            phasecoef)
        det1, det2 = np.triu_indices(self.n_det, 1)  # All possible det pairs
        phase_differences = det_phase[det1] - det_phase[det2]
        time_differences = det_time[det1] - det_time[det2]

        intrinsic = np.concatenate([
            ampcoef[self.n_det:],  # Exclude detector amplitude
            phasecoef[2*self.n_det:]  # Exclude detector phase & time
            ])
        return np.concatenate([[amp_rms],
                               amp_ratios,
                               np.cos(phase_differences),
                               np.sin(phase_differences),
                               time_differences,
                               intrinsic,
                               [np.cos(det_phase[i_refdet]),
                                np.sin(det_phase[i_refdet]),
                                det_time[i_refdet]]])

    def get_transform_kwargs(self, coef, i_refdet, f_ref):
        """
        Return dictionary with the following kwargs, useful to
        instantiate the coordinate transformation:
            * coef0pn
            * phase_refdet_0
            * amp_ref_det
            * t0_refdet
        """
        ampcoef, phasecoef = self.split_amp_phase_coef(coef)
        coef0pn = self.phase_model.get_coef0pn(phasecoef)
        amp_ref_det = ampcoef[i_refdet]

        _, times = self.phase_model.get_detector_phases_and_times(phasecoef)
        t0_refdet = times[i_refdet]

        phase_refdet_0 = self.phase_model(
            np.array([f_ref]), phasecoef)[i_refdet, 0]

        return {'coef0pn': coef0pn,
                'phase_refdet_0': phase_refdet_0,
                'amp_ref_det': amp_ref_det,
                't0_refdet': t0_refdet}

    def guess_shapecoef(self, frequencies, wht_filter,
                        ref_waveform_phase, ref_waveform_amp):
        """
        Find amplitude and phase coefficients that best match a given
        waveform amplitude and phase.

        Return
        ------
        shapecoef: float array
        """
        shapeampcoef_guess \
            = self.amplitude_model.amplitude_tapering.guess_shapeampcoef(
                frequencies, wht_filter, ref_waveform_amp)

        phasecoef_guess = self.phase_model.guess_phasecoef(
            frequencies, ref_waveform_phase)
        shapephasecoef_guess = phasecoef_guess[self.phase_model.n_det:]

        shapecoef_guess = np.concatenate([shapeampcoef_guess,
                                          shapephasecoef_guess])
        return shapecoef_guess


class AmplitudeModel:
    """Simple phenomenological model for the waveform amplitude."""
    def __init__(self, n_det, amplitude_tapering):
        """
        Parameters
        ----------
        n_det: int
            Number of detectors.

        amplitude_tapering: AmplitudeTapering
            Models the merger.
        """
        self.n_det = n_det
        self.amplitude_tapering = amplitude_tapering

    def __call__(self, frequencies, ampcoef, apply_tapering=True):
        """
        Parameters
        ----------
        frequencies: float array of shape (n_frequencies)
            Evaluation frequencies (Hz).

        ampcoef: float array of shape (n_det+1,)
            ampcoef[:n_det] = Amplitude at detector (physical units).
            ampcoef[-1] = Cutoff frequency (Hz).

        apply_tapering: bool
            Whether to model the merger or let the amplitude profile be
            ~ f**(-7/6).

        Return
        ------
        float array of shape (n_det, n_frequencies)
            Waveform amplitude profiles.
        """
        amplitudes = ampcoef[:self.n_det]
        shapeampcoef = ampcoef[self.n_det:]

        # 1e-20 is made up so that `amplitudes` ~ O(1)
        profile = 1e-20 * frequencies ** (-7/6)

        if apply_tapering:
            # tapering -> 1 (f << fcut), -> 0 (f >> fcut)
            profile *= self.amplitude_tapering(frequencies, shapeampcoef)

        return np.outer(amplitudes, profile)

    def get_detector_amp_rms_and_ratios(self, ampcoef):
        """
        Features that encode some extrinsic-parameter information.

        Parameters
        ----------
        ampcoef: float array of shape (n_det+1,)
            ampcoef[:n_det] = Amplitude at detector (physical units).
            ampcoef[-1] = Cutoff frequency (Hz).

        Return
        ------
        amp_rms: float
            Root-mean-square amplitude over detectors.

        amp_ratios: float array of shape (n_det,)
            amp_det / amp_rms
        """
        det_amp = ampcoef[:self.n_det]
        amp_rms = np.linalg.norm(det_amp)
        amp_ratios = det_amp / amp_rms
        return amp_rms, amp_ratios

    @staticmethod
    def _normalize(arr):
        return arr / np.linalg.norm(arr)

    @property
    def n_ampcoef(self):
        """Number of amplitude parameters."""
        return self.n_det + self.amplitude_tapering.n_shapeampcoef


class AmplitudeTapering(utils.NpzMixin):
    """
    Multiplicative correction to A(f) ~ f^{-7/6}.

    Approaches 1 for f << f_merger and 0 for f >> f_merger.
    It is obtained from examples as the mean tapering (over examples),
    plus a correction which is a linear combination of basis functions
    obtained with a singular value decomposition. The tapering is
    aligned so that it happens at a frequency ``fcut``.
    The (log10) cutoff frequency and SVD coefficients are free
    parameters. Their range is computed from the examples and recorded
    as ``.shapeampcoef_bounds``.
    """

    @classmethod
    def from_scratch(cls,
                     waveform_generator,
                     simulation_parameters,
                     frequencies=(1e-2, 1e4, 500),
                     relative_frequencies=(1e-4, 1e1, 1000),
                     n_svd=1,
                     tapering_at_fcut=0.1):
        """
        Parameters
        ----------
        waveform_generator: cogwheel.waveform.WaveformGenerator
            Will be used to generate examples to input into a singular
            value decomposition to construct the model.

        simulation_parameters: pd.DataFrame
            Each row is an example of a binary merger's parameters, per
            `waveform_generator._waveform_params`.

        frequencies: float array or 3-tuple
            Frequency in Hz. If a 3-tuple is passed, it will be unpacked
            into ``np.geomspace`` to create the array.

        relative_frequencies: float array or 3-tuple
            Frequency divided by f_cut. If a 3-tuple is passed, it will
            be unpacked into ``np.geomspace`` to create the array.

        n_svd: int
            How many SVD components to keep in the model for aligned
            taperings.

        tapering_at_fcut: float
            Defines ``fcut`` as the frequency at which the tapering has
            this value.
        """
        if isinstance(frequencies, tuple):
            frequencies = np.geomspace(*frequencies)

        if isinstance(relative_frequencies, tuple):
            relative_frequencies = np.geomspace(*relative_frequencies)

        aligned_taperings, log10fcuts = cls._aligned_taperings_and_log10fcuts(
            waveform_generator, simulation_parameters, frequencies,
            relative_frequencies, tapering_at_fcut)

        mean_aligned_tapering = np.mean(aligned_taperings, axis=0)

        umat, vals, vhmat = np.linalg.svd(
            aligned_taperings - mean_aligned_tapering, full_matrices=False)

        # Only keep desired components
        umat = umat[:, :n_svd]
        vals = vals[:n_svd]
        vhmat = vhmat[:n_svd]

        # Pick sign so that the bases are positive (just to help intuition)
        signs = np.sign(vhmat.sum(axis=1))
        vhmat *= signs[:, np.newaxis]
        umat *= signs

        svd_coefs = umat * vals  # (n_examples, n_svd)
        shapeampcoefs = np.concatenate([log10fcuts[:, np.newaxis], svd_coefs],
                                       axis=1)

        shapeampcoef_bounds = list(zip(shapeampcoefs.min(axis=0),
                                       shapeampcoefs.max(axis=0)))

        return cls(relative_frequencies,
                   mean_aligned_tapering,
                   vhmat,
                   shapeampcoef_bounds,
                   tapering_at_fcut)

    def __init__(self, relative_frequencies, mean_aligned_tapering,
                 vhmat, shapeampcoef_bounds, tapering_at_fcut):
        """
        Generic constructor, normally one would use ``.from_scratch`` or
        ``.from_npz``.

        Parameters
        ----------
        relative_frequencies: (N,) array
            f / f_cut

        mean_aligned_tapering: (N,) array
            Mean tapering evaluated on `relative_frequencies`.

        vhmat: (M, N) array
            SVD basis functions for the departure from the mean
            tapering, evaluated on `relative_frequencies`.

        shapeampcoef_bounds: (M+1, 2) array-like
            Bounds on ``log10_fcut`` and the SVD coefficients.

        tapering_at_fcut: float
            Defines ``fcut`` as the frequency at which the tapering has
            this value.
        """
        self.relative_frequencies = relative_frequencies
        self.mean_aligned_tapering = mean_aligned_tapering
        self.vhmat = vhmat
        self.shapeampcoef_bounds = shapeampcoef_bounds
        self.tapering_at_fcut = tapering_at_fcut

    def __call__(self, frequencies, shapeampcoef):
        """Tapering as a function of frequency."""
        fcut = 10**shapeampcoef[0]
        svd_coef = shapeampcoef[1:]
        aligned_tapering = self.mean_aligned_tapering + svd_coef @ self.vhmat
        tapering = np.interp(frequencies,
                             self.relative_frequencies * fcut,
                             aligned_tapering)
        return tapering

    @property
    def n_shapeampcoef(self):
        """
        Number of shape coefficients, (= n_svd + 1, for the merger
        frequency).
        """
        return len(self.shapeampcoef_bounds)

    def guess_shapeampcoef(self, frequencies, wht_filter, amplitude):
        """
        Estimate of the shape coefficients that approximate the provided
        amplitude profile.

        Parameters
        ----------
        frequencies: float array of shape (n_freq,)
            Frequencies (Hz) at which the amplitude is defined.

        wht_filter: float array of shape(n_det, n_freq)
            Frequency-domain whitening filter in each detector.

        amplitude: float array of shape(n_det, n_freq)
            Frequency domain amplitude profile in each detector.

        Return
        ------
        shapeampcoef: float array
            Contains ``(log10fcut, *svd_coef)``.
        """
        assert wht_filter.shape[-1:] == frequencies.shape
        assert amplitude.shape == wht_filter.shape
        assert wht_filter.ndim == 2

        _, log10fcut = self._aligned_tapering_and_log10fcut(
            frequencies, np.linalg.norm(amplitude, axis=0),
            self.relative_frequencies,
            self.tapering_at_fcut)
        if np.isnan(log10fcut):
            log10fcut = self.shapeampcoef_bounds[0][1]

        fcut = 10 ** log10fcut
        tapering_funcs = np.array(
            [np.interp(frequencies, self.relative_frequencies * fcut, arr)
             for arr in (self.mean_aligned_tapering, *self.vhmat)])
        mat = (np.linalg.norm(wht_filter, axis=0)
               * frequencies**(-7/6)
               * tapering_funcs).T
        target_wht_amp = np.linalg.norm(amplitude * wht_filter, axis=0)
        x = np.linalg.lstsq(mat, target_wht_amp, rcond=None)[0]
        svd_coef = x[1:] / x[0]

        return np.concatenate([[log10fcut], svd_coef])

    @classmethod
    def _aligned_taperings_and_log10fcuts(
            cls, waveform_generator, simulation_parameters, frequencies,
            relative_frequencies, tapering_at_fcut):
        amplitudes = (
            np.abs(waveform_generator.get_hplus_hcross(frequencies,
                                                       par_dic)[0])
            for _, par_dic in simulation_parameters.iterrows())

        aligned_taperings, log10fcuts = zip(
            *(cls._aligned_tapering_and_log10fcut(
                frequencies, amplitude, relative_frequencies,
                tapering_at_fcut)
            for amplitude in amplitudes))

        valid = ~np.isnan(log10fcuts)
        log10fcuts = np.array(log10fcuts)[valid]
        aligned_taperings = np.array(aligned_taperings)[valid]

        return aligned_taperings, log10fcuts

    @staticmethod
    def _aligned_tapering_and_log10fcut(
            frequencies, amplitude, relative_frequencies,
            tapering_at_fcut):
        """
        Tapering as a function of f / fcut, evaluated on
        ``._relative_frequencies``.
        """
        tapering = frequencies**(7/6) * amplitude
        tapering /= tapering[0]

        ind = np.where(tapering > tapering_at_fcut)[0][-1]
        if ind == len(frequencies) - 1:  # Merger frequency too high
            return np.full_like(relative_frequencies, np.nan), np.nan

        inds = [ind + 1, ind]
        fcut = np.interp(tapering_at_fcut,
                         tapering[inds],
                         frequencies[inds])

        aligned_tapering = np.interp(relative_frequencies * fcut,
                                     frequencies,
                                     tapering)

        return aligned_tapering, np.log10(fcut)


class PhaseModel:
    """
    Model the phase profile of the waveform as a function of intrinsic
    parameters in terms of orthogonalized coordinates.
    Shares many similarities with the IAS template bank.

    Interpretation of parameters
    ----------------------------
    pncoef: array of coefficients with physical meaning
        They have analytical expressions.
        * pncoef[:n_det] = phase at each detector
        * pncoef[n_det : 2*n_det] = 2*pi*time at each detector (s)
        * pncoef[2*n_det:] = 0PN, 1PN, 1.5PN prefactors before
            (f/Hz)**pn_exponent

    phasecoef: array of coefficients in an orthonormal basis
        Obtained with a mixture of QR (phase & time part) and SVD
        (intrinsic part). Can be connected to `pncoef` via matrix
        algebra, see `._phasecoef_to_pncoef().

        * phasecoef[:n_det] only affect phase at each detector.
        * phasecoef[n_det : 2*n_det] only affect time and phase at each
          detector.
        * phasecoef[2*n_det:] only affect waveform phase profile
          orthogonal to time and phase shifts.
    """
    # Indices throughout the code refer to array dimensions as follows:
    #     d: detector
    #     f: frequency
    #     n: pncoef
    #     c: phasecoef
    #     i: parameter example
    #     ?: optional dimensions

    _int_pn_exponents = np.array([-5/3, -1, -2/3])

    @classmethod
    def from_scratch(cls,
                     frequencies,
                     fiducial_wht_filter,
                     n_phasecoef=2,
                     mchirp_rng=(1.0, 50.0),
                     q_rng=(0.05, 1.0),
                     n_examples=10**4,
                     pn_phase_tol=0.1,
                     seed=0):
        """
        Parameters
        ----------
        frequencies: (n_freq,) float array
            Frequencies at which the fiducial whitening filter is
            reported (Hz).

        fiducial_wht_filter: (n_det, n_freq) float array
            Fiducial whitening filter used to orthogonalize phase bases.
            E.g. from a `cogwheel.EventData`, but remove the frequencies
            at which it equals 0 like so:
            `event_data.wht_filter[:, event_data.fslice]`.

        n_phasecoef: int
            How many dimensions to use for describing the intrinsic
            parameter space. Increasing this allows more freedom in the
            model.

        mchirp_rng: 2-tuple of floats
            Minimum and maximum chirp-mass with which examples are
            generated (as input to the SVD).

        q_rng: 2-tuple of floats
            Minimum and maximum mass ratio with which examples are
            generated (as input to the SVD).

        n_examples: int
            How many examples to input in the SVD.

        pn_phase_tol: float
            Controls the number of frequencies with which the code will
            work (`.fbin` attribute). A smaller `pn_phase_tol` produces
            finer `fbin`. The user interacts with `fbin` through the
            `.guess_phasecoef()` method.

        seed: {None, int, numpy.random.Generator}
            For reproducible output, since the SVD examples are random.
        """
        rb_splines, weights = cls._get_rbsplines_and_weights(
            frequencies, fiducial_wht_filter, pn_phase_tol)  # f, df
        n_det, _ = weights.shape
        n_ext = 2 * n_det  # phase & time at each detector

        parameter_examples = cls._get_parameter_examples(
            mchirp_rng, q_rng, n_examples, seed)
        intpncoef_examples = cls._parameters_to_intpncoef(
            **parameter_examples)  # ni

        qmat, rmat = cls._get_qr(rb_splines.fbin, weights)  # dfn, nn

        weightedphase_to_pncoef_mat = np.einsum('nN,dfN->ndf',
                                                np.linalg.inv(rmat),
                                                qmat)  # ndf

        # Drop first `n_ext` rows/columns to orthogonalize to phase and time;
        # R is upper triangular so we can drop its first columns too:
        weighted_phase_examples = np.einsum('dfN,Nn,ni->dfi',
                                            qmat[..., n_ext:],
                                            rmat[n_ext:, n_ext:],
                                            intpncoef_examples)  # dfi

        avg_weighted_phase = np.mean(weighted_phase_examples, axis=2)  # df
        avg_pncoef = np.einsum('ndf,df->n',
                               weightedphase_to_pncoef_mat,
                               avg_weighted_phase)  # n
        umat = cls._get_umat(
            weighted_phase_examples - avg_weighted_phase[..., np.newaxis],
            n_phasecoef)  # dfc

        weighted_dphase_basis = np.concatenate(
            [qmat[..., :n_ext], umat], axis=2)  # dfc

        phasecoef_to_dpncoef_mat = np.einsum('ndf,dfc->nc',
                                             weightedphase_to_pncoef_mat,
                                             weighted_dphase_basis)  # nc

        dphase_to_phasecoef_mat = np.einsum('dfc,df->cdf',
                                            weighted_dphase_basis,
                                            weights)  # cdf
        return cls(rb_splines=rb_splines,
                   _dphase_to_phasecoef_mat=dphase_to_phasecoef_mat,
                   _phasecoef_to_dpncoef_mat=phasecoef_to_dpncoef_mat,
                   _avg_pncoef=avg_pncoef)

    def __init__(self,
                 rb_splines,
                 _dphase_to_phasecoef_mat,
                 _phasecoef_to_dpncoef_mat,
                 _avg_pncoef):
        """Generic constructor, use `from_scratch` instead."""
        self.rb_splines = rb_splines
        self._dphase_to_phasecoef_mat = _dphase_to_phasecoef_mat  # cdf
        self._phasecoef_to_dpncoef_mat = _phasecoef_to_dpncoef_mat  # nc
        self._avg_pncoef = _avg_pncoef  # n

    def __call__(self, frequencies, phasecoef):
        """
        Return waveform phase.

        Parameters
        ----------
        frequencies: float array
            Frequencies in Hz.

        phasecoef: float array
            Coefficients of the orthogonal basis.

        Return
        ------
        phase: float array of shape (n_det, n_freq)
        """
        pnphases = self._get_pnphases(frequencies, self.n_det)  # dfn
        pncoef = self._phasecoef_to_pncoef(phasecoef)  # n
        return pnphases @ pncoef  # df

    @property
    def n_det(self):
        """Number of detectors."""
        _, n_det, _ = self._dphase_to_phasecoef_mat.shape
        return n_det

    @property
    def n_phasecoef(self):
        """Number of phase parameters."""
        return self._dphase_to_phasecoef_mat.shape[0]

    def get_detector_phases_and_times(self, phasecoef):
        """
        Return
        ------
        phases: (n_det,) float array
            Waveform phase at each detector (rad).

        times: (n_det,) float array
            Arrival time at rach detector (s).
        """
        pncoef = self._phasecoef_to_pncoef(phasecoef)
        phases = pncoef[: self.n_det] % (2*np.pi)
        times = -pncoef[self.n_det : 2*self.n_det] / (2*np.pi)
        return phases, times

    def get_coef0pn(self, phasecoef):
        """
        Return estimate of the 0PN coefficient.

        Multiply this by (f/Hz)^{-5/3} to get the phase evolution.
        """
        pncoef = self._phasecoef_to_pncoef(phasecoef)
        return pncoef[2 * self.n_det]

    def guess_phasecoef(self, frequencies, phase):
        """
        Compute the coefficients of the orthonormal phase bases that
        produce the best (least-squares) approximation to a target phase
        profile.

        Parameters
        ----------
        phase: (n_det, n_freq) float array
            Target phase profile, must be evaluated at ``.fbin``.
        """
        assert phase.shape == (self.n_det, len(frequencies))

        phase_fbin = scipy.interpolate.make_interp_spline(
            frequencies, phase, axis=1, k=1
            )(self.rb_splines.fbin)

        avg_phase = (self._get_pnphases(self.rb_splines.fbin, self.n_det)
                     @ self._avg_pncoef)  # df
        dphase = phase_fbin - avg_phase  # df
        phasecoef = np.einsum('cdf,df->c',
                              self._dphase_to_phasecoef_mat,
                              dphase)
        return phasecoef

    def detphasecoef(self, det_phase):
        """
        Return value of ``phasecoef[:n_det]`` that will produce a given
        detector phase, i.e. ``pncoef[:n_det]``.
        """
        # Note: relies on the orthogonality of the detector phase
        # coefficients to the remaining ones.
        return np.linalg.inv(
            self._phasecoef_to_dpncoef_mat[:self.n_det, :self.n_det]
            ) @ det_phase

    def _phasecoef_to_pncoef(self, phasecoef):
        return self._avg_pncoef + self._phasecoef_to_dpncoef_mat @ phasecoef

    @staticmethod
    def _get_rbsplines_and_weights(frequencies, fiducial_wht_filter,
                                   pn_phase_tol):
        rb_splines = RelativeBinningSplines(frequencies,
                                            pn_phase_tol=pn_phase_tol)
        weights_f = frequencies**(-7/6) * fiducial_wht_filter
        weights_fbin = np.sqrt(np.abs(
            rb_splines.get_summary_weights(weights_f**2)))
        # (take abs because spline interpolation can produce values < 0)

        weights_fbin /= np.linalg.norm(weights_fbin)
        return rb_splines, weights_fbin

    @classmethod
    def _get_qr(cls, frequencies, weights):
        """QR decomposition of the matrix of weighted phases."""
        n_det, n_freq = weights.shape
        pnphases = cls._get_pnphases(frequencies, n_det)  # dfn
        weighted_phases = np.einsum('df,dfn->dfn', weights, pnphases)  # dfn

        # For the purposes of inner products, it may be useful to think
        # of the detector and frequency axes as a single big axis:
        reshaped_weighted_phases = weighted_phases.reshape(n_det*n_freq, -1)
        reshaped_qmat, rmat = unique_qr(reshaped_weighted_phases)  # (df)n, nn

        qmat = reshaped_qmat.reshape(n_det, n_freq, -1)  # dfn
        return qmat, rmat  # dfn, nn

    @classmethod
    def _get_pnphases(cls, frequencies, n_det):
        """
        Basis functions of the PN expansion.

        float array of shape (n_det, n_freq, n_pncoef).
        """
        n_freq = len(frequencies)
        n_ext = 2 * n_det  # phase at detector, time at detector
        n_int = len(cls._int_pn_exponents)
        extrinsic_phases = np.einsum('nf,dD->dfnD',
                                     (np.ones(n_freq), frequencies),
                                     np.eye(n_det)
                                    ).reshape((n_det, n_freq, n_ext))  # dfn
        intrinsic_phases = np.broadcast_to(
            np.power.outer(frequencies, cls._int_pn_exponents),
            (n_det, n_freq, n_int))  # dfn

        return np.concatenate([extrinsic_phases, intrinsic_phases],
                              axis=2)  # dfn

    @staticmethod
    def _get_parameter_examples(mchirp_rng, q_rng, n_examples, seed):
        """
        Return dict with arrays of samples of ``m1, m2, s1z, s2z``.
        They are distributed uniformly in mchirp^(-5/3), eta, s1z, s2z
        according to a scrambled Halton sequence.
        """
        mchirp_53_rng = np.power(mchirp_rng, -5/3)[::-1]
        eta_rng = cogwheel.gw_utils.q_to_eta(np.asarray(q_rng))
        s_rng = (-1, 1)
        l_bounds, u_bounds = zip(mchirp_53_rng, eta_rng, s_rng, s_rng)
        mchirp_53, eta, s1z, s2z = qmc.scale(
            qmc.Halton(4, seed=seed).random(n_examples), l_bounds, u_bounds).T
        mchirp = mchirp_53 ** (-3/5)
        m1, m2 = cogwheel.gw_utils.mchirpeta_to_m1m2(mchirp, eta)
        return {'m1': m1, 'm2': m2, 's1z': s1z, 's2z': s2z}

    @staticmethod
    def _parameters_to_intpncoef(m1, m2, s1z, s2z):
        """
        Return post-Newtonian coefficients such that the intrinsic-
        parameter part of the frequency-domain waveform phase is
            phase = pnphases @ intpncoef.
        """
        mtot = m1 + m2
        eta = m1 * m2 / mtot**2
        chis = (s1z + s2z) / 2
        chia = (s1z - s2z) / 2
        delta = (m1 - m2) / mtot
        beta = 113/12 * (chis + delta*chia - 76/113*eta*chis)

        pncoef = np.array([
            -3/128 / eta * (np.pi*lal.MTSUN_SI*mtot)**(-5/3),
            -3/128 * (55/9 + 3715/756/eta) / (np.pi*lal.MTSUN_SI*mtot),
            -3/128 * (4*beta-16*np.pi)/eta * (np.pi*lal.MTSUN_SI*mtot)**(-2/3),
            ])  # n?

        return pncoef

    @staticmethod
    def _get_umat(weighted_dphase_examples,
                  n_phasecoef):
        """
        Return set of basis functions that describe the contribution to
        the phase from intrinsic-parameters, found from examples by SVD.

        These examples and hence the basis functions are assumed to have
        been orthogonalized to phase and time offsets in each detector.
        Note that a different phase and time shift is applied in each
        detector in order to orthogonalize the phases.
        """
        n_det, n_freq, n_examples = weighted_dphase_examples.shape

        reshaped_weighted_dphase_examples = np.reshape(
            weighted_dphase_examples, (n_det*n_freq, n_examples))  # (df)i

        reshaped_umat = np.linalg.svd(
            reshaped_weighted_dphase_examples, full_matrices=False
            )[0][:, :n_phasecoef]  # Drop least important dimensions  # (df)c

        return reshaped_umat.reshape(n_det, n_freq, n_phasecoef)  # dfc


def unique_qr(mat):
    """QR decomposition ensuring R has positive diagonal elements."""
    qmat, rmat = np.linalg.qr(mat)
    signs = np.diagflat(np.sign(np.diag(rmat)))
    return qmat @ signs, signs @ rmat
