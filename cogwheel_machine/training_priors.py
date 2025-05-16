"""Prior classes intended to generate simulation parameters."""
import numpy as np

from cogwheel.prior import (
    Prior,
    UniformPriorMixin,
    IdentityTransformMixin,
    FixedPrior)

import cogwheel.utils
from cogwheel.gw_prior.combined import (
    UniformLuminosityVolumePrior,
    RegisteredPriorMixin,
    CombinedPrior,
    IsotropicInclinationPrior,
    IsotropicSkyLocationPrior,
    UniformTimePrior,
    UniformPhasePrior,
    UniformPolarizationPrior,
    UniformEffectiveSpinPrior,
    ZeroInplaneSpinsPrior,
    ZeroTidalDeformabilityPrior,
    FixedReferenceFrequencyPrior)

from . import transform


# ----------------------------------------------------------------------
# Modular priors:

class ZeroAlignedSpinsPrior(FixedPrior):
    """Set inplane spins to zero."""
    standard_par_dic = {'s1z': 0.,
                        's2z': 0.,}


class LogMassPrior(UniformPriorMixin, Prior):
    """
    Flat in log mchirp, log q.

    Auxiliary prior intended for generating training parameters.
    It tries to achieve a balance between a "geometric" prior in
    which each waveform shape is equally represented, and a
    "physical" prior.
    """
    standard_params = ['m1', 'm2']
    range_dic = {'lnmchirp': None,
                 'lnq': None}

    def __init__(self, *, mchirp_range, q_min, **kwargs):
        lnq_min = np.log(q_min)
        self.range_dic = {'lnmchirp': np.log(mchirp_range),
                          'lnq': (lnq_min, 0)}
        super().__init__(**kwargs)

    @staticmethod
    def transform(lnmchirp, lnq):
        """(mchirp, lnq) to (m1, m2)."""
        q = np.exp(-np.abs(lnq))
        mchirp = np.exp(lnmchirp)
        m1 = mchirp * (1 + q)**.2 / q**.6
        return {'m1': m1,
                'm2': m1 * q}

    @staticmethod
    def inverse_transform(m1, m2):
        """(m1, m2) to (mchirp, lnq)."""
        q = m2 / m1
        mchirp = m1 * q**.6 / (1 + q)**.2
        return {'lnmchirp': np.log(mchirp),
                'lnq': np.log(q)}

    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return {'mchirp_range': np.exp(self.range_dic['lnmchirp']),
                'q_min': np.exp(self.range_dic['lnq'][0])}

    def ln_jacobian_determinant(self, m1, m2):
        """
        Return log of the Jacobian determinant of `.transform`.

        I.e.: log|∂{lnmchirp, lnq} / ∂{m1, m2}|
        """
        return -np.log(m1 * m2)


class UniformAmplitudePrior(UniformPriorMixin, Prior):
    """Distance prior uniform in amp_refdet ≡ 1/d_hat."""
    range_dic = {'amp_refdet': None}
    standard_params = ['d_luminosity']
    conditioned_on = ['ra', 'dec', 'psi', 'iota', 'm1', 'm2']

    def __init__(self, tgps, ref_det_name, d_hat_min, d_hat_max,
                 **kwargs):
        self.range_dic = {'amp_refdet': (1 / d_hat_max, 1 / d_hat_min)}

        self.aux_prior = UniformLuminosityVolumePrior(
            tgps=tgps, ref_det_name=ref_det_name)

        super().__init__(tgps=tgps,
                         ref_det_name=ref_det_name,
                         d_hat_min=d_hat_min,
                         d_hat_max=d_hat_max,
                         **kwargs)

    def transform(self, amp_refdet, ra, dec, psi, iota, m1, m2) -> dict:
        """amp_refdet to d_luminosity"""
        d_hat = 1 / amp_refdet

        return self.aux_prior.transform(d_hat, ra, dec, psi, iota, m1, m2)

    def inverse_transform(self, d_luminosity, ra, dec, psi, iota,
                          m1, m2) -> dict:
        """d_luminosity to amp_refdet"""
        d_hat = self.aux_prior.inverse_transform(
            d_luminosity, ra, dec, psi, iota, m1, m2)['d_hat']

        return {'amp_refdet': 1 / d_hat}

    def ln_jacobian_determinant(self, d_luminosity, ra, dec, psi, iota,
                                m1, m2) -> float:
        """
        Return log of the Jacobian determinant of `.transform`.

        I.e.: log|∂{amp_refdet} / ∂{d_luminosity}|
        """
        amp_refdet = self.inverse_transform(
            d_luminosity, ra, dec, psi, iota, m1, m2)['amp_refdet']
        return np.log(amp_refdet / d_luminosity)

    def get_init_dict(self):
        """Keyword arguments to reproduce the class instance."""
        return {'tgps': self.aux_prior.tgps,
                'ref_det_name': self.aux_prior.ref_det_name,
                'd_hat_max': 1 / self.range_dic['amp_refdet'][0],
                'd_hat_min': 1 / self.range_dic['amp_refdet'][1]}



class UniformDHatPrior(UniformPriorMixin, UniformLuminosityVolumePrior):
    """
    Auxiliary prior intended for generating training parameters.
    Flat in `d_hat` (https://arxiv.org/pdf/2207.03508#equation.3.18).
    """


class PhasePrior(UniformPriorMixin, IdentityTransformMixin, Prior):
    """Uniform prior for the phase. No change of coordinates."""
    # The reason why this class is used here instead of
    # cogwheel.gw_prior.UniformPhasePrior is that UniformPhasePrior
    # would default to `phase_refdet_0 = 0` during training, and a
    # different phase_refdet_0 during post-processing, giving
    # conflicting values for phi_ref_hat. In this class there is no
    # phi_ref_hat.
    range_dic = {'phi_ref': (0, 2*np.pi)}


# ----------------------------------------------------------------------
# Combine the modular priors:

class NoSpinTrainingPrior(RegisteredPriorMixin,
                          CombinedPrior):
    """Intended for generating training parameters."""
    prior_classes = [LogMassPrior,
                     IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     PhasePrior,
                     UniformAmplitudePrior,
                     ZeroAlignedSpinsPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]

    default_transform_class = transform.TargetSpaceTransformNoSpins


class AlignedSpinTrainingPrior(RegisteredPriorMixin,
                               CombinedPrior):
    """Intended for generating training parameters."""
    prior_classes = cogwheel.utils.replace(NoSpinTrainingPrior.prior_classes,
                                           ZeroAlignedSpinsPrior,
                                           UniformEffectiveSpinPrior)

    default_transform_class = transform.TargetSpaceTransformAlignedSpins


class AlignedSpinSamplingPrior(RegisteredPriorMixin, CombinedPrior):
    """Intended for sampling, to test the amortized inference."""
    prior_classes = cogwheel.utils.replace(
        AlignedSpinTrainingPrior.prior_classes,
        PhasePrior,
        UniformPhasePrior)


class AlignedSpinUniformDHatTrainingPrior(RegisteredPriorMixin,
                                          CombinedPrior):
    """Intended for generating training parameters."""
    prior_classes = cogwheel.utils.replace(
        AlignedSpinTrainingPrior.prior_classes,
        UniformAmplitudePrior,
        UniformDHatPrior)

    default_transform_class = transform.TargetSpaceTransformAlignedSpins
