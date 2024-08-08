"""
Settings for the training that need to be shared across modules.
"""
import numpy as np
from cogwheel_machine.training_priors import NoSpinTrainingPrior
from cogwheel_machine.transform import TargetSpaceTransform


# Fiducial reference time, not the time of any actual event.
# (The training should use Earth-fixed coordinates. But since the
# existing implementations use ra, dec, we hack them by using a fiducial
# GPS time until we have a cleaner implementation.)
TGPS = 0.0

PRIOR_KWARGS = {
    'mchirp_range': (1., 50.),
    'detector_pair': 'HL',
    'tgps': TGPS,
    'ref_det_name': 'L',
    'f_avg': 100.,
    'f_ref': 100.,
    'd_hat_max': 400
}

EVENT_DATA_KWARGS = {
    'eventname': None,
    'duration': 32,
    'detector_names': 'HL',
    'asd_funcs': ['asd_H_O3', 'asd_L_O3'],
    'tgps': TGPS,
    'tcoarse': 0.,
    }

PN_PHASE_TOL_COMPRESSION = 0.1
N_COHERENT_SEGMENTS = 8

N_SIMULATIONS = 10**2  # Increase for real-life usage!

PRIOR_CLASS = NoSpinTrainingPrior

TRANSFORM_CLASS = TargetSpaceTransform

APPROXIMANT = 'IMRPhenomD'

MASK_CONDITIONS = [('snr0', np.greater, 8),
                   ('snr0', np.less, 50),
                  ]

# ----------------------------------------------------------------------
# Training

# kwargs to sbi.utils.posterior_nn
POSTERIOR_NN_KWARGS = {'model': 'nsf',
                       'hidden_features': 256}

# kwargs to sbi.inference.SNPE.train
TRAIN_KWARGS = {'training_batch_size': 8192,
                'stop_after_epochs': 20,
                'learning_rate': 1e-3,
                'show_train_summary': True}

MAX_TRAINING_EXAMPLES = None  # int
DEVICE = 'cuda'
