"""
Phenomenological waveform model that works with coordinates that are
approximately orthonormal (under a reference PSD).
"""
import argparse
from collections import OrderedDict
from pathlib import Path
import scipy.interpolate
from scipy.stats import qmc
import numpy as np

import lal

import cogwheel.data
import cogwheel.gw_utils
import cogwheel.waveform

from .rbsplines import RelativeBinningSplines
from . import condor_utils, hdf5_utils, utils


class PhenomenologicalWaveformGenerator(hdf5_utils.HDF5Mixin):
    """
    A simple waveform model to find a reference waveform quickly.

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
    def from_rundir(cls, rundir):
        """
        Attempt to load instance, else construct and save it.

        Parameters
        ----------
        rundir : os.PathLike
            Path to a directory on which ``generate_parameters`` has
            been run.
        """
        rundir = Path(rundir)

        filename = rundir/utils.WAVEFORM_MODEL_FILENAME

        if filename.exists():
            return hdf5_utils.read_hdf5(filename)

        config = utils.load_data_config(rundir)
        dummy_data = cogwheel.data.EventData.gaussian_noise(
            **config.EVENT_DATA_KWARGS)

        frequencies = dummy_data.frequencies[dummy_data.fslice]
        wht_filter = dummy_data.wht_filter[:, dummy_data.fslice]

        waveform_model = cls.from_waveforms(
            frequencies,
            wht_filter,
            config.PN_PHASE_TOL,
            config.INCLUDE_GLOBAL_PHASE_AND_TIME,
        )

        waveform_model.to_hdf5(filename)
        return waveform_model

    @classmethod
    def from_waveforms(cls, frequencies, fiducial_wht_filter,
                       pn_phase_tol, include_global_phase_and_time=False):
        """
        Parameters
        ----------
        frequencies : (n_freq,) float array

        fiducial_wht_filter : (n_det, n_freq) float array

        pn_phase_tol : float
            Determines the internal frequency resolution at which the
            phase model will compute inner products in order to
            orthogonalize the phase bases. Lower tolerance means higher
            resolution.

        include_global_phase_and_time : bool
            Whether to include the global time and phase of arrival in
            the processed_coef (summary data passed to the network).
            Excluding them (False) enforces time and phase shifts to be
            exact equivariances of the model, which may improve
            performance and reduce overfitting.

        See Also
        --------
        .from_rundir
        """
        amplitude_model = AmplitudeModel(len(fiducial_wht_filter))
        phase_model = PhaseModel.from_scratch(
            frequencies=frequencies,
            fiducial_wht_filter=fiducial_wht_filter,
            pn_phase_tol=pn_phase_tol)
        return cls(amplitude_model, phase_model, include_global_phase_and_time)

    def __init__(self, amplitude_model, phase_model,
                 include_global_phase_and_time=True):
        """
        Parameters
        ----------
        amplitude_model : AmplitudeModel
            Phenomenological model for the amplitude of the waveform.

        phase_model : PhaseModel
            Phenomenological model for the phase of the waveform.

        include_global_phase_and_time : bool
            Whether to include the global time and phase of arrival in
            the processed_coef (summary data passed to the network).
            Excluding them (False) enforces time and phase shifts to be
            exact equivariances of the model, which may improve
            performance and reduce overfitting.
        """
        assert phase_model.n_det == amplitude_model.n_det
        self.amplitude_model = amplitude_model
        self.phase_model = phase_model
        self.include_global_phase_and_time = include_global_phase_and_time

    def __call__(self, frequencies, coef):
        """
        Parameters
        ----------
        frequencies : float array of shape (n_frequencies,)
            Evaluation frequencies (Hz).

        coef : float array of shape (`.n_coef`,)
            ampcoef, phasecoef concatenated.

        Returns
        -------
        complex array of shape (n_det, n_frequencies)
            Waveform at detectors.
        """
        assert coef.shape == (self.n_coef,)
        ampcoef, phasecoef = self.split_amp_phase_coef(coef)
        amplitude = self.amplitude_model(frequencies, ampcoef)
        phase = self.phase_model(frequencies, phasecoef)
        return amplitude * np.exp(1j*phase)

    def split_amp_phase_coef(self, coef):
        """Return `ampcoef`, `phasecoef` from `coef`."""
        assert coef.shape == (self.n_coef,)
        return np.split(coef, [self.amplitude_model.n_ampcoef])

    def waveform_fiducial_amp_and_phase(self, frequencies, shapecoef):
        """
        Similar to `.__call__` but it takes fewer parameters, and sets
        detector amplitudes to 1 and detector phases to 0.

        (Useful because these can be maximized analytically.)

        Parameters
        ----------
        frequencies : float array of shape (n_frequencies,)
            Evaluation frequencies (Hz).

        shapecoef : float array of shape (`.n_coef` - 2*n_det,)
            ampcoef, phasecoef concatenated, but with entries
            corresponding to detector amplitude and phase removed.

        Returns
        -------
        complex array of shape (n_det, n_frequencies)
            Waveform at detectors.
        """
        coef = self.coef_from_shapecoef(shapecoef,
                                        det_amp=np.ones(self.n_det),
                                        det_phase=np.zeros(self.n_det))
        return self(frequencies, coef)

    def coef_from_shapecoef(self, shapecoef, det_amp, det_phase):
        """
        Insert `det_amp` and `det_phase` into `shapecoef`.

        Insert overall amplitude at each detector and phase at each
        detector into the array of shape coefficients, to generate a
        complete array of coefficients.

        Parameters
        ----------
        shapecoef : float array
            Waveform parameters other than detectors' amplitude and
            phase.

        det_amp : float array of shape (n_det,)
            Overall amplitude at each detector.

        det_phase : float array of shape (n_det,)
            Overall phase at each detector.

        Returns
        -------
        coef : float array
        """
        assert shapecoef.shape == (self.n_coef - 2*self.n_det,)

        n_shapeampcoef = self.amplitude_model.n_ampcoef - self.n_det
        shapeampcoef, shapephasecoef = np.split(shapecoef, [n_shapeampcoef])
        ampcoef = (det_amp, shapeampcoef)
        phasecoef = (self.phase_model.detphasecoef(det_phase), shapephasecoef)
        return np.concatenate((*ampcoef, *phasecoef))

    @property
    def n_coef(self):
        """Number of phenomenological parameters, ``coef``."""
        return self.amplitude_model.n_ampcoef + self.phase_model.n_phasecoef

    @property
    def n_det(self):
        """Number of detectors."""
        return self.amplitude_model.n_det

    def process_coef(self, coef, i_refdet, f_cut):
        """
        Transform `coef` to make it more suitable for a neural network.

        Parameters
        ----------
        coef : float array of shape (`.n_coef`,)
            ampcoef, phasecoef concatenated.

        Returns
        -------
        float32 array
            Concatenation of the following quantities:

            ============================  ============================
            Quantity                      Shape
            ============================  ============================
            amp                       1
            amp_ratios                    n_det
            cos(phase_differences)        n_det * (n_det - 1) / 2
            sin(phase_differences)        n_det * (n_det - 1) / 2
            time_differences              n_det * (n_det - 1) / 2
            amplitude shape parameters    n_shapeampcoef
            phase shape parameters        n_intphasecoef
            log10f_cut                    1
            cos(ref_det_phase) (*)        1
            sin(ref_det_phase) (*)        1
            ref_det_time       (*)        1
            ============================  ============================

            (*) only if `self.include_global_phase_and_time` is True.
        """
        ampcoef, phasecoef = self.split_amp_phase_coef(coef)
        amp, amp_ratios = self.amplitude_model.get_detector_amp_and_ratios(
            ampcoef)

        det_phase, det_time = self.phase_model.get_detector_phases_and_times(
            phasecoef)
        det1, det2 = np.triu_indices(self.n_det, 1)  # All possible det pairs
        phase_differences = det_phase[det1] - det_phase[det2]
        time_differences = det_time[det1] - det_time[det2]

        features = [
            [amp],
            amp_ratios,
            np.cos(phase_differences),
            np.sin(phase_differences),
            time_differences,
            ampcoef[self.n_det:],  # Exclude detector amplitude
            phasecoef[2*self.n_det:],  # Exclude detector phase & time
            [np.log10(f_cut)],
        ]
        if self.include_global_phase_and_time:
            features.extend([[
                np.cos(det_phase[i_refdet]),
                np.sin(det_phase[i_refdet]),
                det_time[i_refdet],
            ]])

        return np.concatenate(features)

    @property
    def processed_coef_keys(self):
        """Bookkeeping of what the entries in processed_coef mean."""
        det_pairs = list(zip(*np.triu_indices(self.n_det, 1)))
        keys = ['amp',
                *(f'amp_ratio_{i}' for i in range(self.n_det)),
                *(f'cos_phase_difference_{j}-{i}' for i, j in det_pairs),
                *(f'sin_phase_difference_{j}-{i}' for i, j in det_pairs),
                *(f'time_difference_{j}-{i}' for i, j in det_pairs),
                *self.amplitude_model.ampcoef_keys[self.n_det:],
                *self.phase_model.phasecoef_keys[2*self.n_det:],
                'log10_fcut',
                ]
        if self.include_global_phase_and_time:
            keys.extend([
                'cos_refdet_phase',
                'sin_refdet_phase',
                'refdet_time'
            ])
        return keys

    def get_transform_kwargs(self, coef, i_refdet, f_ref):
        """
        Return kwargs to instantiate the coordinate transformation.

        Namely:

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

        phase_refdet_0 = (
            self.phase_model(np.array([f_ref]), phasecoef)[i_refdet, 0]
            + 3/4 * np.pi  # See arxiv.org/pdf/gr-qc/0509116 Eq (3.4)
        ) % (2*np.pi)

        return {'coef0pn': coef0pn,
                'phase_refdet_0': phase_refdet_0,
                'amp_ref_det': amp_ref_det,
                't0_refdet': t0_refdet}

    def guess_shapecoef(self, frequencies, waveform_phase):
        """
        Find coefficients that best match a given waveform phase.

        Returns
        -------
        float array
        """
        return self.phase_model.guess_phasecoef(frequencies, waveform_phase
                                               )[self.phase_model.n_det:]



class AmplitudeModel(hdf5_utils.HDF5Mixin):
    """Simple phenomenological model for the waveform amplitude."""
    def __init__(self, n_det):
        """
        Parameters
        ----------
        n_det : int
            Number of detectors.

        amplitude_tapering : AmplitudeTapering
            Models the merger.
        """
        self.n_det = n_det

    def __call__(self, frequencies, ampcoef):
        """
        Parameters
        ----------
        frequencies : float array of shape (n_frequencies)
            Evaluation frequencies (Hz).

        ampcoef : float array of shape (n_det,)
            Amplitudes at each detector (physical units).

        Returns
        -------
        float array of shape (n_det, n_frequencies)
            Waveform amplitude profiles.
        """

        # 1e-20 is made up so that `amplitudes` ~ O(1)
        profile = 1e-20 * frequencies ** (-7/6)

        return np.outer(ampcoef, profile)

    def get_detector_amp_and_ratios(self, ampcoef):
        """
        Features that encode some extrinsic-parameter information.

        Parameters
        ----------
        ampcoef : float array of shape (n_det,)
            Amplitudes at each detector (physical units).

        Returns
        -------
        amp : float
            Quadrature-sum amplitude over detectors.

        amp_ratios : ndarray of shape (n_det,)
            Detector amplitudes normalized by amp.
        """
        amp = np.linalg.norm(ampcoef)
        amp_ratios = ampcoef / amp
        return amp, amp_ratios

    @staticmethod
    def _normalize(arr):
        return arr / np.linalg.norm(arr)

    @property
    def n_ampcoef(self):
        """Number of amplitude parameters."""
        return self.n_det

    @property
    def ampcoef_keys(self):
        """List of names describing the entries in ``ampcoef``."""
        return [f'amp_{i}' for i in range(self.n_det)]


class PhaseModel(hdf5_utils.HDF5Mixin):
    """
    Model the phase profile of the waveform as a function of intrinsic
    parameters in terms of orthogonalized coordinates.
    Shares many similarities with the IAS template bank.

    Interpretation of parameters
    ----------------------------
    pncoef : array of coefficients with physical meaning
        They have analytical expressions.

        * pncoef[:n_det] = phase at each detector
        * pncoef[n_det : 2*n_det] = 2*pi*time at each detector (s)
        * pncoef[2*n_det:] = 0PN, 1PN, 1.5PN prefactors before
            (f/Hz)^pn_exponent

    phasecoef : array of coefficients in an orthonormal basis
        Obtained with a mixture of QR (phase & time part) and SVD
        (intrinsic part). Can be connected to `pncoef` via matrix
        algebra, see `._phasecoef_to_pncoef()`.

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

    _cache = OrderedDict()
    _cache_size = 2

    @classmethod
    def from_scratch(cls,
                     frequencies,
                     fiducial_wht_filter,
                     n_intphasecoef=2,
                     mchirp_rng=(1.0, 50.0),
                     q_rng=(0.05, 1.0),
                     n_examples=10**4,
                     pn_phase_tol=0.1,
                     seed=0):
        """
        Parameters
        ----------
        frequencies : (n_freq,) float array
            Frequencies at which the fiducial whitening filter is
            reported (Hz).

        fiducial_wht_filter : (n_det, n_freq) float array
            Fiducial whitening filter used to orthogonalize phase bases.
            E.g. from a `cogwheel.EventData`, but remove the frequencies
            at which it equals 0 like so:
            `event_data.wht_filter[:, event_data.fslice]`.

        n_intphasecoef : int
            How many dimensions to use for describing the intrinsic
            parameter space. Increasing this allows more freedom in the
            model.

        mchirp_rng : 2-tuple of floats
            Minimum and maximum chirp-mass with which examples are
            generated (as input to the SVD).

        q_rng : 2-tuple of floats
            Minimum and maximum mass ratio with which examples are
            generated (as input to the SVD).

        n_examples : int
            How many examples to input in the SVD.

        pn_phase_tol : float
            Controls the number of frequencies with which the code will
            work (`.fbin` attribute). A smaller `pn_phase_tol` produces
            finer `fbin`. The user interacts with `fbin` through the
            `.guess_phasecoef()` method.

        seed : {None, int, numpy.random.Generator}
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
            n_intphasecoef)  # dfc

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
        self._det_phase_to_detphasecoef_mat = np.linalg.inv(
            self._phasecoef_to_dpncoef_mat[:self.n_det, :self.n_det])  # dd

    def __call__(self, frequencies, phasecoef):
        """
        Return waveform phase.

        Parameters
        ----------
        frequencies : float array
            Frequencies in Hz.

        phasecoef : float array
            Coefficients of the orthogonal basis.

        Returns
        -------
        phase : float array of shape (n_det, n_freq)
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

    @property
    def phasecoef_keys(self):
        """List of names describing the entries in ``phasecoef``."""
        n_intphasecoef = self.n_phasecoef - 2*self.n_det
        return [*(f'detphasecoef_{i}' for i in range(self.n_det)),
                *(f'dettimecoef_{i}' for i in range(self.n_det)),
                *(f'intphasecoef_{i}' for i in range(n_intphasecoef))]

    def get_detector_phases_and_times(self, phasecoef):
        """
        Returns
        -------
        phases : (n_det,) float array
            Waveform phase at each detector (rad).

        times : (n_det,) float array
            Arrival time at each detector (s).
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
        phase : (n_det, n_freq) float array
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
        return self._det_phase_to_detphasecoef_mat @ det_phase

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

        Returns
        -------
        float array of shape (n_det, n_freq, n_pncoef).
        """
        frequencies = np.asarray(frequencies)
        cache_key = (frequencies.tobytes(), frequencies.shape,
                     frequencies.dtype, n_det)
        try:
            return cls._cache[cache_key]
        except KeyError:
            pass

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
        pnphases = np.concatenate([extrinsic_phases, intrinsic_phases],
                                  axis=2)[()]  # dfn

        cls._cache[cache_key] = pnphases
        if len(cls._cache) > cls._cache_size:  # Delete oldest cache
            cls._cache.popitem(last=False)

        return pnphases

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
    def _get_umat(weighted_dphase_examples, n_phasecoef):
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


def setup_condor_sub(rundir, request_memory='8G', request_disk='1G',
                     submit=False, **submit_kwargs):
    """
    Create a script to run the waveform_model job on HTCondor.

    This will generate the following files:
        {rundir}/submission_scripts/waveform_model.{sub,sh}

    Parameters
    ----------
    rundir : os.PathLike
        Simulations directory, on which `generate_data` has already
        been run.

    Returns
    -------
    pathlib.Path
        Path to the HTCondor submit file.
    """
    rundir = Path(rundir).resolve()
    stem = rundir/'submission_scripts'/'waveform_model'
    module = 'labrador.waveform_model'
    return condor_utils.setup_condor_sub(stem, module,
                                         request_memory=request_memory,
                                         request_disk=request_disk,
                                         submit=submit,
                                         arguments=rundir,
                                         **submit_kwargs)


def main(rundir):
    """
    Create a file ``{rundir}/waveform_model.h5``.

    Parameters
    ----------
    rundir : os.PathLike
        Run directory, must contain a training a test directories with
        simulation parameters (after ``generate_parameters.py`` has been
        run).
    """
    PhenomenologicalWaveformGenerator.from_rundir(rundir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create a file `{rundir}/waveform_model.h5`.')
    parser.add_argument('rundir', help='path to a run directory.')

    main(**vars(parser.parse_args()))
