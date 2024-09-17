"""Prior classes intended to generate simulation parameters."""
import numpy as np

from cogwheel.prior import (
    Prior,
    UniformPriorMixin,
    IdentityTransformMixin,
    FixedPrior)

from cogwheel.gw_prior.combined import (
    UniformLuminosityVolumePrior,
    RegisteredPriorMixin,
    CombinedPrior,
    IsotropicInclinationPrior,
    IsotropicSkyLocationPrior,
    UniformTimePrior,
    UniformPolarizationPrior,
    UniformEffectiveSpinPrior,
    ZeroInplaneSpinsPrior,
    ZeroTidalDeformabilityPrior,
    FixedReferenceFrequencyPrior)

import cogwheel.utils

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
                     UniformDHatPrior,
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
