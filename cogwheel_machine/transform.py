"""
Define ``TargetSpaceTransform``, a class that implements a
coordinate transformation that give a first approximation to the
normalizing flow.
"""
import numpy as np

from cogwheel.prior import Prior, CombinedPrior
from cogwheel import gw_utils
from cogwheel import gw_prior


class TransformMixin:
    """
    Indicates that a `Prior` object is intended to be used only for
    its ``transform`` and ``inverse_transform`` methods.
    """
    def lnprior(self, *args, **kwargs):
        """Intentionally not implemented."""
        del args, kwargs
        raise NotImplementedError(
            'This class is intended to be used for its `transform` '
            'and `inverse_transform` only.')


class MassesTransform(TransformMixin, Prior):
    """
    Coordinate transformation for the masses, in which the posterior
    should be almost independent of the data.

    This is achieved by having a guess of the chirp mass (based on
    the reference waveform) and also reparametrizing the chirp mass
    so that the uncertainties are homogeneous.
    The coordinate associated to chirp-mass is
    `diff_reparametrized_mchirp`, which is the difference between the
    reparametrized chirp mass and the reparametrized chirp mass guess.
    The posterior for this should resemble a Gaussian centered at 0.
    """
    range_dic = {'diff_reparametrized_mchirp': (np.nan, np.nan),
                 'lnq': (np.nan, np.nan)}
    standard_params = ['m1', 'm2']

    def __init__(self, mchirp_guess, **kwargs):
        """
        Parameters
        ----------
        mchirp_guess: float
            Estimate of the chirp-mass (Msun).
        """
        super().__init__(**kwargs)
        self.mchirp_guess = mchirp_guess
        self._mchirp_reparametrizer = gw_utils._ChirpMassRangeEstimator()
        self._reparametrized_mchirp_guess = self._reparametrized_mchirp(
            mchirp_guess)

    def transform(self, diff_reparametrized_mchirp, lnq):
        """``sampled_params`` to ``inverse_params``."""
        reparametrized_mchirp = (self._reparametrized_mchirp_guess
                                 + diff_reparametrized_mchirp)
        mchirp = self._mchirp(reparametrized_mchirp)

        q = np.exp(-np.abs(lnq))
        m1 = mchirp * (1 + q)**.2 / q**.6
        return {'m1': m1,
                'm2': m1 * q}

    def inverse_transform(self, m1, m2):
        """``inverse_params`` to ``sampled_params``."""
        q = m2 / m1
        mchirp = m1 * q**.6 / (1 + q)**.2

        reparametrized_mchirp = self._reparametrized_mchirp(mchirp)
        diff_reparametrized_mchirp = (reparametrized_mchirp
                                      - self._reparametrized_mchirp_guess)
        return {'diff_reparametrized_mchirp': diff_reparametrized_mchirp,
                'lnq': np.log(q)}

    def _reparametrized_mchirp(self, mchirp):
        return self._mchirp_reparametrizer._x_of_mchirp(mchirp)

    def _mchirp(self, reparametrized_mchirp):
        return self._mchirp_reparametrizer._mchirp_of_x(reparametrized_mchirp)

    def get_init_dict(self):
        """Keyword arguments to reproduce the class instance."""
        return {'mchirp_guess': self.mchirp_guess}


class PhaseTransform(TransformMixin, gw_prior.UniformPhasePrior):
    def __init__(self, *, tgps, ref_det_name, f_avg, phase_refdet_0,
                 **kwargs):
        """
        Parameters
        ----------
        tgps: float
            Fiducial GPS time used in the training set.
            NOT the real GPS time of the event!

        ref_det_name: str
            Reference detector name, e.g. 'H' for Hanford.

        f_avg: float
            Fiducial f_avg used in the training set.

        phase_refdet_0: float
            Phase of the reference waveform at the reference detector.
        """
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, f_avg=f_avg,
                         phase_refdet_0=phase_refdet_0, **kwargs)
        self._phase_refdet_0 = phase_refdet_0 % (2*np.pi)

    def get_init_dict(self):
        """Keyword arguments to reproduce the class instance."""
        init_dict = super().get_init_dict()
        del init_dict['par_dic_0']
        init_dict['phase_refdet_0'] = self._phase_refdet_0
        return init_dict


class TimeTransform(TransformMixin, Prior):
    standard_params = ['t_geocenter']
    range_dic = {'dt_refdet': (np.nan, np.nan)}
    conditioned_on = ['ra', 'dec']

    def __init__(self, *, tgps, ref_det_name, t0_refdet, **kwargs):
        """
        Parameters
        ----------
        tgps: float
            Fiducial GPS time used in the training set.
            NOT the real GPS time of the event!

        ref_det_name: str
            Reference detector name, e.g. 'H' for Hanford.

        amp_ref_det: float
            Amplitude of the reference waveform at the reference
            detector (units don't matter as long as they are
            consistent across training and production).
        """
        super().__init__(tgps=tgps, ref_det_name=ref_det_name,
                         t0_refdet=t0_refdet, **kwargs)
        self.t0_refdet = t0_refdet
        self._time_transformer = gw_prior.UniformTimePrior(
            tgps=tgps,
            ref_det_name=ref_det_name,
            t0_refdet=np.nan, dt0=np.nan)

    def transform(self, dt_refdet, ra, dec):
        """``sampled_params`` to ``inverse_params``."""
        t_refdet = self.t0_refdet + dt_refdet
        return self._time_transformer.transform(t_refdet, ra, dec)

    def inverse_transform(self, t_geocenter, ra, dec):
        """``inverse_params`` to ``sampled_params``."""
        t_refdet = self._time_transformer.inverse_transform(
            t_geocenter, ra, dec)['t_refdet']
        return {'dt_refdet': t_refdet - self.t0_refdet}

    def get_init_dict(self):
        """Keyword arguments to reproduce the class instance."""
        init_dict = self._time_transformer.get_init_dict()
        return {'tgps': init_dict['tgps'],
                'ref_det_name': init_dict['ref_det_name'],
                't0_refdet': self.t0_refdet}


class DistanceTransform(TransformMixin, Prior):
    standard_params = ['d_luminosity']
    range_dic = {'relative_dhat': (np.nan, np.nan)}
    conditioned_on = ['ra', 'dec', 'psi', 'iota', 'm1', 'm2']

    def __init__(self, *, tgps, ref_det_name, amp_ref_det, **kwargs):
        """
        Parameters
        ----------
        tgps: float
            Fiducial GPS time used in the training set.
            NOT the real GPS time of the event!

        ref_det_name: str
            Reference detector name, e.g. 'H' for Hanford.

        amp_ref_det: float
            Amplitude of the reference waveform at the reference
            detector (units don't matter as long as they are
            consistent across training and production).
        """
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, **kwargs)
        self.amp_ref_det = amp_ref_det

        self._distance_transformer = gw_prior.UniformLuminosityVolumePrior(
            tgps=tgps,
            ref_det_name=ref_det_name)

    def transform(self, relative_dhat, ra, dec, psi, iota, m1, m2):
        """``sampled_params`` to ``inverse_params``."""
        d_hat = relative_dhat / self.amp_ref_det
        return self._distance_transformer.transform(
            d_hat, ra, dec, psi, iota, m1, m2)

    def inverse_transform(self, d_luminosity, ra, dec, psi, iota, m1, m2):
        """``inverse_params`` to ``sampled_params``."""
        d_hat = self._distance_transformer.inverse_transform(
            d_luminosity, ra, dec, psi, iota, m1, m2)['d_hat']
        return {'relative_dhat': d_hat * self.amp_ref_det}

    def get_init_dict(self):
        """Keyword arguments to reproduce the class instance."""
        init_dict = self._distance_transformer.get_init_dict()
        return {'tgps': init_dict['tgps'],
                'ref_det_name': init_dict['ref_det_name'],
                'amp_ref_det': self.amp_ref_det}


class TargetSpaceTransform(CombinedPrior):
    prior_classes = [MassesTransform,
                     gw_prior.IsotropicInclinationPrior,
                     gw_prior.UniformPolarizationPrior,
                     gw_prior.IsotropicSkyLocationPrior,
                     TimeTransform,
                     PhaseTransform,
                     DistanceTransform
                     ]
