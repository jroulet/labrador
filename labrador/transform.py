"""
Define ``TargetSpaceTransform``, a class that implements a coordinate
transformation that gives a first approximation to the normalizing flow.
"""
import numpy as np

import lal

from cogwheel.prior import Prior, CombinedPrior, UnitJacobianMixin
from cogwheel import gw_prior
from . import pn_coords

# pylint: disable=arguments-differ


class TransformMixin:
    """
    Indicates that a `Prior` object is intended to be used only for its
    ``transform`` and ``inverse_transform`` methods.
    """
    def lnprior(self, *args, **kwargs):
        """Intentionally not implemented."""
        del self, args, kwargs
        raise NotImplementedError(
            'This class is intended to be used for its `transform` and '
            '`inverse_transform` only.')


# ----------------------------------------------------------------------
# Modular transforms for few parameters at a time


class MassesTransform(TransformMixin, Prior):
    """
    Coordinate transformation for the masses, in which the posterior
    should be almost independent of the data.

    This is achieved by reparametrizing the chirp mass in terms of the
    0-pN coefficient, so that the uncertainties are homogeneous.
    The 0-pN coefficient $-3 / 128 (pi mchirp Hz)^{-5/3}$ has a maximum
    value of 0, attained for high mchirp (for which the pN is not valid
    in the detector band). Thus we regularize it by smoothly switching
    to a linear relation above `mchirp_break`.
    The coordinate associated to chirp-mass is `diff_regularized0pn`,
    which is the difference between the regularized 0pN coefficient and
    our guess for it from the reference waveform.
    To the extent that the 0pN term is a good description of the
    waveform, the posterior for this quantity should resemble a Gaussian
    centered at 0.
    """
    range_dic = {'diff_regularized0pn': (-np.inf, np.inf),
                 'lnq': None}
    standard_params = ['m1', 'm2']

    def __init__(self, coef0pn, q_min, mchirp_break=60.0, **kwargs):
        """
        Parameters
        ----------
        coef0pn : float
            Estimate of the 0-pN coefficient from the reference
            waveform.

        mchirp_break : float
            Chirp mass (Msun) at which to shift from the post-Newtonian
            regime to a linear regime for the chirp-mass
            reparametrization.

        See Also
        --------
        waveform_model.PhenomenologicalWaveformGenerator.get_transform_kwargs
        waveform_model.PhaseModel.get_coef0pn
        """
        self.range_dic = self.range_dic | {'lnq': (np.log(q_min), 0.0)}
        super().__init__(**kwargs)
        self.mchirp_break = mchirp_break
        self.coef0pn = coef0pn

    def transform(self, diff_regularized0pn, lnq):
        """``sampled_params`` to ``inverse_params``."""
        regularized0pn = self.coef0pn + diff_regularized0pn
        mchirp = self._mchirp(regularized0pn)

        q = np.exp(-np.abs(lnq))
        m1 = mchirp * (1 + q)**.2 / q**.6
        return {'m1': m1,
                'm2': m1 * q}

    def inverse_transform(self, m1, m2):
        """``inverse_params`` to ``sampled_params``."""
        q = m2 / m1
        mchirp = m1 * q**.6 / (1 + q)**.2

        regularized0pn = self._regularized0pn(mchirp)
        diff_regularized0pn = regularized0pn - self.coef0pn
        return {'diff_regularized0pn': diff_regularized0pn,
                'lnq': np.log(q)}

    def get_init_dict(self):
        """Keyword arguments to reproduce the class instance."""
        return {'coef0pn': self.coef0pn,
                'mchirp_break': self.mchirp_break,
                'q_min': np.exp(self.range_dic['lnq'][0])}

    def _regularized0pn(self, mchirp):
        mchirp = np.asarray(mchirp)  # piecewise needs arrays
        return np.piecewise(mchirp,
                            [mchirp < self.mchirp_break],
                            [self._regularized0pn_low,
                             self._regularized0pn_high])[()]

    def _regularized0pn_low(self, mchirp):
        return -3 / 128 * (np.pi*mchirp*lal.MTSUN_SI)**(-5/3)

    def _regularized0pn_high(self, mchirp):
        return (1/128 * (np.pi*self.mchirp_break*lal.MTSUN_SI)**(-5/3)
                * (5*mchirp / self.mchirp_break - 8))

    def _mchirp(self, regularized0pn):
        regularized0pn = np.asarray(regularized0pn)  # piecewise needs arrays
        boundary = self._regularized0pn(self.mchirp_break)
        return np.piecewise(regularized0pn,
                            [regularized0pn < boundary],
                            [self._mchirp_low, self._mchirp_high])[()]

    def _mchirp_low(self, regularized0pn):
        return (-128/3*regularized0pn) ** (-3/5) / (np.pi*lal.MTSUN_SI)

    def _mchirp_high(self, regularized0pn):
        return self.mchirp_break / 5 * (
            128*(np.pi*lal.MTSUN_SI*self.mchirp_break)**(5/3) * regularized0pn
            + 8)

    def ln_jacobian_determinant(self, m1, m2):
        """
        Return log of the Jacobian determinant between sampled and
        standard parameters.

        I.e.
            ln(|∂{diff_regularized0pn, lnq} / ∂{m1, m2}|)
        """
        q = m2 / m1
        mchirp = m1 * q**.6 / (1 + q)**.2
        regularized0pn = self._regularized0pn(mchirp)

        if mchirp < self.mchirp_break:
            lnj_regularized0pn_lnmchirp = np.log(np.abs(5/3*regularized0pn))
        else:
            lnj_regularized0pn_lnmchirp = np.log(np.abs(
                regularized0pn
                + 1/16 * (np.pi*self.mchirp_break*lal.MTSUN_SI)**(-5/3)))

        lnj_lnmchirplnq_m1m2 = -np.log(m1*m2)

        return lnj_regularized0pn_lnmchirp + lnj_lnmchirplnq_m1m2


class PhaseTransform(TransformMixin, gw_prior.UniformPhasePrior):
    """
    Coordinate transformation for the orbital phase.

    The coordinate is cogwheel's ``phi_ref_hat`` except the baseline
    phase ``phi_refdet_0` is passed by the user.
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
        init_dict = super().get_init_dict()
        del init_dict['par_dic_0']
        init_dict['phase_refdet_0'] = self._phase_refdet_0
        return init_dict


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

    The coordinate is `relative_dhat`, i.e. cogwheel's d_hat divided by
    a fiducial d_hat.
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

        I.e.
            ln(|∂{relative_dhat} / ∂{d_luminosity}|)
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


class TargetSpaceTransformNoSpins(CombinedPrior):
    """Full coordinate transformation for all waveform parameters."""
    prior_classes = [MassesTransform,
                     gw_prior.IsotropicInclinationPrior,
                     gw_prior.UniformPolarizationPrior,
                     gw_prior.IsotropicSkyLocationPrior,
                     TimeTransform,
                     PhaseTransform,
                     DistanceTransform,
                     ]


class TargetSpaceTransformAlignedSpins(CombinedPrior):
    """Full coordinate transformation for all waveform parameters."""
    prior_classes = [*TargetSpaceTransformNoSpins.prior_classes,
                     gw_prior.UniformEffectiveSpinPrior,
                     ]


class _PNCoordinatesPrior(gw_prior.PNCoordinatesPrior):
    range_dic = {'mu1': (-np.inf, np.inf),
                 'mu2': (-np.inf, np.inf),
                 'lnq': None,
                 's2z': (-1, 1),
                }
    def __init__(self, eigvecs=None, **kwargs):
        # TODO; for now just put some values for par_dic_0 and eigvecs
        if eigvecs is None:
            eigvecs = np.array([[-1.57616411, -0.04111396],
                                [-0.54265283,  0.08432735],
                                [-0.27537869,  0.06914793]])

        par_dic_0 = dict.fromkeys(['m1', 'm2', 's1z', 's2z'], 1.0)

        super().__init__(eigvecs=eigvecs, par_dic_0=par_dic_0, **kwargs)

        # The parent class tries to be smart about the range_dic, undo.
        # TODO change cogwheel.gw_prior.PNCoordinatesPrior, perhaps allow
        # par_dic_0 = None
        # Perhaps make a base class with abstract standard_lnprior
        self.range_dic.update(mu1=(-np.inf, np.inf),
                              mu2=(-np.inf, np.inf))
        self.cubemin = np.array([rng[0] for rng in self.range_dic.values()])
        cubemax = np.array([rng[1] for rng in self.range_dic.values()])
        self.cubesize = cubemax - self.cubemin
        self.folded_cubesize = self.cubesize.copy()
        self.folded_cubesize[self._folded_inds] /= 2

    def get_init_dict(self):
        """Return kwargs to reproduce this class instance."""
        # We don't want to pollute the .json with the dummy par_dic_0
        init_dict = super().get_init_dict()
        del init_dict['par_dic_0']
        return init_dict


class TargetSpaceTransformAlignedSpinsPN(CombinedPrior):
    prior_classes = [_PNCoordinatesPrior,
                     gw_prior.IsotropicInclinationPrior,
                     gw_prior.UniformPolarizationPrior,
                     gw_prior.IsotropicSkyLocationPrior,
                     TimeTransform,
                     PhaseTransform,
                     DistanceTransform,
                     ]


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
