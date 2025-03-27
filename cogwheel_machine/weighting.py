"""Compute weights to go from a simulation prior to a physical prior."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cogwheel_machine import utils
from cogwheel_machine.prior_ratio import PriorRatio


def get_weights_filename(physical_prior_class) -> str:
    """Standard name for the weights file."""
    return f'weights_{physical_prior_class.__name__}'


def main(rundir, recompute_existing=False):
    """
    Generate files with weights in the training and test directories.

    Parameters
    ----------
    rundir : os.PathLike
        Run directory, must contain a training a test directories with
        simulation parameters (after ``generate_parameters.py`` has been
        run).

    recompute_existing : bool
        If a weights file already exists, we only recompute it if this
        flag is True (default False).
    """
    rundir = Path(rundir)
    data_config = utils.load_data_config(rundir)

    # TODO rename PRIOR_CLASS -> SIMULATION_PRIOR_CLASS
    simulation_prior = data_config.PRIOR_CLASS(**data_config.PRIOR_KWARGS)

    for prior_cls in data_config.PHYSICAL_PRIOR_CLASSES:
        physical_prior = prior_cls(**data_config.PRIOR_KWARGS)
        prior_ratio = PriorRatio(physical_prior, simulation_prior)
        get_ln_prior_ratio = np.vectorize(prior_ratio.ln_prior_ratio)

        for datadir in rundir/utils.TRAINING_DIR, rundir/utils.TEST_DIR:
            filename = datadir/get_weights_filename(prior_cls)
            if not recompute_existing and filename.exists():
                print(f'Skipping existing {filename}...')
                continue

            parameters = pd.read_feather(datadir/utils.PARAMETERS_FILENAME
                                        )[simulation_prior.standard_params]

            weights = np.exp(get_ln_prior_ratio(**parameters))
            np.save(filename, weights)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Compute weights to turn samples from a simulation prior
                       into a physical prior.''')
    parser.add_argument('rundir', help='path to a run directory.')
    parser.add_argument('--recompute_existing', action='store_true',
                        help='Recompute existing files.',)

    main(**vars(parser.parse_args()))
