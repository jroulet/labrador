"""
Settings for the training that need to be shared across modules.

TODO: This could be automatically saved in the simulation directory?
Maybe turn into a JSONMixin?
"""

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
    'f_avg': 50.,
    'f_ref': 50.,
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
