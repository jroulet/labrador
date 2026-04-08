"""
Define analytical coordinate transformations to make the posterior
approximately Gaussian.
"""
import numpy as np

from cogwheel.prior import Prior, CombinedPrior, UnitJacobianMixin
from cogwheel import gw_prior
from . import pn_coords

# pylint: disable=arguments-differ


class TransformMixin:
    """
    Indicates that a `Prior` object is intended to be used only for its
    :py:meth:`transform` and :py:meth:`inverse_transform` methods.
    """
    def lnprior(self, *args, **kwargs):
        """Intentionally not implemented."""
        del self, args, kwargs
        raise NotImplementedError(
            'This class is intended to be used for its `transform` and '
            '`inverse_transform` only.')


# ----------------------------------------------------------------------
# Modular transforms for few parameters at a time

class PhaseTransform(TransformMixin, gw_prior.UniformPhasePrior):
    """
    Coordinate transformation for the orbital phase.

    The coordinate is cogwheel's ``phi_ref_hat`` except the baseline
    phase ``phi_refdet_0`` is passed by the user.
    """
    def __init__(self, *, tgps, ref_det_name, f_avg, phase_refdet_0,
                 **kwargs):
        """
        Parameters
        ----------
        tgps : float
            Fiducial GPS time used in the training set.
            NOT the real GPS time of the event!

        ref_det_name : str
            Reference detector name, e.g. 'H' for Hanford.

        f_avg : float
            Fiducial f_avg used in the training set.

        phase_refdet_0 : float
            Phase of the reference waveform at the reference detector.
        """
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, f_avg=f_avg,
                         phase_refdet_0=phase_refdet_0, **kwargs)
        self._phase_refdet_0 = phase_refdet_0 % (2*np.pi)

    def get_init_dict(self):
        """Keyword arguments to reproduce the class instance."""
        return super().get_init_dict(phase_refdet_0=self._phase_refdet_0)


class TimeTransform(TransformMixin, UnitJacobianMixin, Prior):
    """
    Coordinate transformation for the geocenter time of arrival.

    The coordinate is the arrival time at the reference detector, minus
    a fiducial arrival time at the reference detector.
    """
    standard_params = ['t_geocenter']
    range_dic = {'dt_refdet': (-np.inf, np.inf)}
    conditioned_on = ['ra', 'dec']

    def __init__(self, *, tgps, ref_det_name, t0_refdet, **kwargs):
        """
        Parameters
        ----------
        tgps : float
            Fiducial GPS time used in the training set.
            NOT the real GPS time of the event!

        ref_det_name : str
            Reference detector name, e.g. 'H' for Hanford.

        amp_ref_det : float
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
    """
    Coordinate transformation for the distance.

    The coordinate is ``relative_dhat``, i.e. cogwheel's ``d_hat``
    divided by a fiducial ``d_hat``.
    """
    standard_params = ['d_luminosity']
    range_dic = {'relative_dhat': (0.0, np.inf)}
    conditioned_on = ['ra', 'dec', 'psi', 'iota', 'm1', 'm2']

    def __init__(self, *, tgps, ref_det_name, amp_ref_det, **kwargs):
        """
        Parameters
        ----------
        tgps : float
            Fiducial GPS time used in the training set.
            NOT the real GPS time of the event!

        ref_det_name : str
            Reference detector name, e.g. 'H' for Hanford.

        amp_ref_det : float
            Amplitude of the reference waveform at the reference
            detector (units don't matter as long as they are
            consistent across training and production).
        """
        super().__init__(tgps=tgps, ref_det_name=ref_det_name,
                         amp_ref_det=amp_ref_det, **kwargs)
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

    def ln_jacobian_determinant(self, d_luminosity, ra, dec, psi, iota,
                                m1, m2):
        """
        Return log of the Jacobian determinant between sampled and
        standard parameters.

        I.e.::

            ln |∂{relative_dhat} / ∂{d_luminosity}|
        """
        lnj_relativedhat_dhat = np.log(self.amp_ref_det)

        lnj_dhat_dluminosity \
            = self._distance_transformer.ln_jacobian_determinant(
                d_luminosity, ra, dec, psi, iota, m1, m2)

        return lnj_relativedhat_dhat + lnj_dhat_dluminosity

    def get_init_dict(self):
        """Keyword arguments to reproduce the class instance."""
        init_dict = self._distance_transformer.get_init_dict()
        return {'tgps': init_dict['tgps'],
                'ref_det_name': init_dict['ref_det_name'],
                'amp_ref_det': self.amp_ref_det}


# ----------------------------------------------------------------------
# Combined transforms for the full parameter space


class TargetSpaceTransformAlignedSpinsPN2(CombinedPrior):
    prior_classes = [pn_coords.PNCoordinatesPrior2,
                     gw_prior.IsotropicInclinationPrior,
                     gw_prior.UniformPolarizationPrior,
                     gw_prior.IsotropicSkyLocationPrior,
                     TimeTransform,
                     PhaseTransform,
                     DistanceTransform,
                     ]


class BasicAlignedSpinsTransform(CombinedPrior):
    """Use simply `mchirp`, `lnq` as mass coordinates."""
    prior_classes = [gw_prior.UniformDetectorFrameMassesPrior,
                     gw_prior.UniformEffectiveSpinPrior,
                     gw_prior.IsotropicInclinationPrior,
                     gw_prior.UniformPolarizationPrior,
                     gw_prior.IsotropicSkyLocationPrior,
                     TimeTransform,
                     PhaseTransform,
                     DistanceTransform,
                     ]
