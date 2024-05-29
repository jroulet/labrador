"""
Phenomenological waveform model that works with coordinates
that are approximately orthonormal (under a reference PSD).
"""
import scipy.linalg
import scipy.optimize
from scipy.stats import qmc
import numpy as np

import lal

import cogwheel.gw_utils

from .rbsplines import RelativeBinningSplines


class PhenomenologicalWaveformGenerator:
    @classmethod
    def from_event_data(cls, event_data, pn_phase_tol):
        frequencies = event_data.frequencies[event_data.fslice]

        amplitude_model = AmplitudeModel(len(event_data.detector_names))
        phase_model = PhaseModel.from_scratch(
            frequencies=frequencies,
            fiducial_wht_filter=event_data.wht_filter[:, event_data.fslice],
            pn_phase_tol=pn_phase_tol)
        return cls(amplitude_model, phase_model)

    def __init__(self, amplitude_model, phase_model):
        assert phase_model.n_det == amplitude_model.n_det
        self.amplitude_model = amplitude_model
        self.phase_model = phase_model

    def __call__(self, frequencies, coef):
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
        ampcoef, phasecoef = np.split(coef, [self.amplitude_model.n_ampcoef])
        amplitude = self.amplitude_model(frequencies, ampcoef)
        phase = self.phase_model(frequencies, phasecoef)
        return amplitude * np.exp(1j*phase)

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
        return self.amplitude_model.n_ampcoef + self.phase_model.n_phasecoef

    @property
    def n_det(self):
        return self.amplitude_model.n_det

    def get_geometry_features(self, coef):
        """
        Parameters
        ----------
        coef: float array of shape (`.n_coef`,)
            ampcoef, phasecoef concatenated.

        Return
        ------
        float array with summary quantities related to arrival amplitude,
        phase and time, that encode the extrinsic parameters of the source.
        """
        ampcoef, phasecoef = np.split(coef, [self.amplitude_model.n_ampcoef])
        amp_rms, amp_ratios = self.amplitude_model.get_detector_amp_rms_and_ratios(ampcoef)
        phase_differences, time_differences \
            = self.phase_model.get_detector_phase_and_time_differences(phasecoef)
        return np.concatenate([[amp_rms],
                               amp_ratios,
                               np.cos(phase_differences),
                               np.sin(phase_differences),
                               time_differences])


class AmplitudeModel:
    """Simple phenomenological model for the waveform amplitude."""
    def __init__(self, n_det, tapering_width=0.1):
        self.n_det = n_det
        self.tapering_width = tapering_width

    def __call__(self, frequencies, ampcoef):
        """
        Parameters
        ----------
        frequencies: float array of shape (n_frequencies)
            Evaluation frequencies (Hz).

        ampcoef: float array of shape (n_det+1,)
            ampcoef[:n_det] = Amplitude at detector (physical units).
            ampcoef[-1] = Cutoff frequency (Hz).

        Return
        ------
        float array of shape (n_det, n_frequencies)
            Waveform amplitude profiles.
        """
        amplitudes = ampcoef[:self.n_det]
        log10_fcut = ampcoef[self.n_det]

        # tapering -> 1 (f << fcut), -> 0 (f >> fcut)
        tapering = self._sigmoid(
            (log10_fcut - np.log10(frequencies)) / self.tapering_width)

        profile = frequencies ** (-7/6) * tapering
        return np.outer(amplitudes, profile)

    def guess_log10_fcut(self, frequencies, wht_filter, amplitude):
        """
        Estimate of the log10 of the cutoff frequency of the provided
        amplitude profile.

        Parameters
        ----------
        frequencies: float array of shape(n_freq)
            Frequencies (Hz) at which the amplitude and whitening filter
            are defined.

        wht_filter: float array of shape(n_det, n_freq)
            Frequency-domain whitening filter in each detector.

        amplitude: float array of shape(n_det, n_freq)
            Frequency domain amplitude profile in each detector.

        Return
        ------
        log10_fcut: float
            Estimate of the base-10 logarithm of the cutoff frequency of
            the provided amplitude profile.
        """
        assert wht_filter.shape[-1:] == frequencies.shape
        assert amplitude.shape == wht_filter.shape

        target_wht_amp = self._normalize(amplitude * wht_filter)

        def objective(log10_fcut):
            ampcoef = np.concatenate([np.ones(self.n_det), [log10_fcut]])
            wht_amp = self._normalize(self(frequencies, ampcoef) * wht_filter)
            return -np.tensordot(target_wht_amp, wht_amp)

        return scipy.optimize.minimize_scalar(objective, bounds=[1., 3.5]).x

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

    @staticmethod
    def _sigmoid(value):
        return 1 / (1 + np.exp(-value))

    @property
    def n_ampcoef(self):
        return self.n_det + 1


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
    _max_mchirp_guess = 50.0

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
        fbin, weights = cls._get_fbin_and_weights(
            frequencies, fiducial_wht_filter, pn_phase_tol)  # f, df
        n_det, _ = weights.shape
        n_ext = 2 * n_det  # phase & time at each detector

        parameter_examples = cls._get_parameter_examples(
            mchirp_rng, q_rng, n_examples, seed)
        intpncoef_examples = cls._parameters_to_intpncoef(
            **parameter_examples)  # ni

        qmat, rmat = cls._get_qr(fbin, weights)  # dfn, nn

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
        return cls(fbin=fbin,
                   _dphase_to_phasecoef_mat=dphase_to_phasecoef_mat,
                   _phasecoef_to_dpncoef_mat=phasecoef_to_dpncoef_mat,
                   _avg_pncoef=avg_pncoef)

    @classmethod
    def from_npz(cls, filename):
        raise NotImplementedError

    def __init__(self,
                 fbin,
                 _dphase_to_phasecoef_mat,
                 _phasecoef_to_dpncoef_mat,
                 _avg_pncoef):
        """
        Generic constructor, use `from_npz` or `from_scratch` instead.
        """
        self._fbin = fbin  # f
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
    def fbin(self):
        """
        Frequencies at which the weights to ortogonalize coordinates
        were computed. Users should not modify this once created.
        """
        return self._fbin

    @property
    def n_det(self):
        _, n_det, _ = self._dphase_to_phasecoef_mat.shape
        return n_det

    @property
    def n_phasecoef(self):
        return self._dphase_to_phasecoef_mat.shape[0]

    def get_detector_phases_and_times(self, phasecoef):
        """Return array of `t2-t1` for each pair of detectors."""
        pncoef = self._phasecoef_to_pncoef(phasecoef)
        phases = pncoef[: self.n_det] % (2*np.pi)
        times = -pncoef[self.n_det : 2*self.n_det] / (2*np.pi)
        return phases, times

    def guess_mchirp(self, phasecoef):
        """Return estimate of chirp mass (Msun)."""
        pncoef = self._phasecoef_to_pncoef(phasecoef)
        coef_0pn = pncoef[2 * self.n_det]

        if coef_0pn < 0:
            # Due to noise, the best fit `phasecoef` may be unphysical
            # i.e. would produce `mchirp**(-5/3) < 0`
            return self._max_mchirp_guess

        mchirp = (-128/3*pncoef[ind_0pn]) ** (-3/5) / (np.pi*lal.MTSUN_SI)
        return min(mchirp, self._max_mchirp_guess)

    def guess_phasecoef(self, phase):
        """
        Compute the coefficients of the orthonormal phase bases that
        produce the best (least-squares) approximation to a target phase
        profile.

        Parameters
        ----------
        phase: (n_det, n_freq) float array
            Target phase profile, must be evaluated at ``.fbin``.
        """
        assert phase.shape == (self.n_det, len(self.fbin))

        avg_phase = (self._get_pnphases(self.fbin, self.n_det)
                     @ self._avg_pncoef)  # df
        dphase = phase - avg_phase  # df
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

    def get_detector_phase_and_time_differences(self, phasecoef):
        """
        Return detector phase differences and time differences, which
        should contain most of the information about sky location.

        Return
        ------
        phase_differences: float array of length `n_det * (n_det-1) / 2`
            Arrival phase difference in each pair of detectors.

        time_differences: float array of length `n_det * (n_det-1) / 2`
            Arrival time difference in each pair of detectors.
        """
        pncoef = self._phasecoef_to_dpncoef_mat @ phasecoef
        det_phase, det_time, _ = np.split(pncoef, [self.n_det, 2*self.n_det])
        det1, det2 = np.triu_indices(self.n_det, 1)  # All possible detector pairs
        phase_differences = det_phase[det1] - det_phase[det2]
        time_differences = det_time[det1] - det_time[det2]
        return phase_differences, time_differences

    def _phasecoef_to_pncoef(self, phasecoef):
        return self._avg_pncoef + self._phasecoef_to_dpncoef_mat @ phasecoef

    @staticmethod
    def _get_fbin_and_weights(frequencies, fiducial_wht_filter,
                              pn_phase_tol):
        rb_splines = RelativeBinningSplines(frequencies,
                                            pn_phase_tol=pn_phase_tol)
        weights_f = frequencies**(-7/6) * fiducial_wht_filter
        weights_fbin = np.sqrt(np.abs(
            rb_splines.get_summary_weights(weights_f**2)))
        # (take abs because spline interpolation can produce values < 0)

        weights_fbin /= np.linalg.norm(weights_fbin)
        return rb_splines.fbin, weights_fbin

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

        reshaped_umat = scipy.linalg.svd(
            reshaped_weighted_dphase_examples, full_matrices=False
            )[0][:, :n_phasecoef]  # Drop least important dimensions  # (df)c

        return reshaped_umat.reshape(n_det, n_freq, n_phasecoef)  # dfc


def unique_qr(mat):
    """QR decomposition ensuring R has positive diagonal elements."""
    qmat, rmat = scipy.linalg.qr(mat, mode='economic')
    signs = np.diagflat(np.sign(np.diag(rmat)))
    return qmat @ signs, signs @ rmat
