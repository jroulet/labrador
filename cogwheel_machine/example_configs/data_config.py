"""
Settings for generating the training and testing sets, that need to be
shared across modules.
"""
import numpy as np
from cogwheel_machine import training_priors


# Fiducial reference time, not the time of any actual event.
# (The training should use Earth-fixed coordinates. But since the
# existing implementations use ra, dec, we hack them by using a fiducial
# GPS time until we have a cleaner implementation.)
TGPS = 0.0

PRIOR_KWARGS = {
    'mchirp_range': (1., 50.),
    'q_min': 1/20,
    'detector_pair': 'HL',
    'tgps': TGPS,
    'ref_det_name': 'L',
    'f_avg': 100.,
    'f_ref': 100.,
    'd_hat_max': 400.,
}

EVENT_DATA_KWARGS = {
    'eventname': None,
    'duration': 32,
    'detector_names': 'HL',
    'asd_funcs': ['asd_H_O3', 'asd_L_O3'],
    'tgps': TGPS,
    'tcoarse': 0.,
}

PN_PHASE_TOL = 0.1

# ``PN_PHASE_TOL_COMPRESSION = None`` makes PN_PHASE_TOL_COMPRESSION
# equal to PN_PHASE_TOL, but simulations are faster if it is ``None``.
PN_PHASE_TOL_COMPRESSION = None

N_COHERENT_SEGMENTS = 8

N_TRAINING_SIMULATIONS = 10**2  # Increase for real-life usage!
N_TEST_SIMULATIONS = 10**2
QMC = True

PRIOR_CLASS = training_priors.AlignedSpinTrainingPrior

TRANSFORM_CLASS = PRIOR_CLASS.default_transform_class

APPROXIMANT = 'IMRPhenomD'

MASK_CONDITIONS = [
    ('snr0', np.greater, 8),
    ('snr0', np.less, 50),
]
