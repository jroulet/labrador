"""Settings for the inference (physical) prior."""
from pathlib import Path
import cogwheel
from labrador import utils


def _get_data_config_prior_kwargs():
    # Note that this file should reside in priordir (within rundir).
    rundir = Path(__file__).resolve().parent.parent
    data_config = utils.load_data_config(rundir)
    return data_config.PRIOR_KWARGS


PRIOR_CLASS = cogwheel.gw_prior.AlignedSpinIASPrior
PRIOR_KWARGS = _get_data_config_prior_kwargs()
PRIOR = PRIOR_CLASS(**PRIOR_KWARGS)
