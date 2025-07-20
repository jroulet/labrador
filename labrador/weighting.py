"""Compute weights to go from a simulation prior to a physical prior."""
import argparse
import os
from pathlib import Path

import xgboost
import numpy as np
import pandas as pd
import cogwheel.utils

from scipy.optimize import differential_evolution

from cogwheel.prior_ratio import PriorRatio

from . import utils


LN_PRIOR_RATIOS_FILENAME = 'ln_prior_ratios.npy'


def _compute_ln_prior_ratios(rundir, recompute_existing=False):
    """
    Generate files with log prior ratios in the training and test
    directories.

    The files contain a log(physical_prior / simulation_prior) for each
    simulation. There is a separate file for each physical prior in
    data_config.py, and for the training and test data.

    Parameters
    ----------
    rundir : os.PathLike
        Run directory, must contain a training a test directories with
        simulation parameters (after ``generate_parameters.py`` has been
        run).

    recompute_existing : bool
        If a ln_prior_ratios file already exists, we only recompute it
        if this flag is True (default False).
    """
    rundir = Path(rundir)
    data_config = utils.load_data_config(rundir)

    # TODO rename PRIOR_CLASS -> SIMULATION_PRIOR_CLASS
    simulation_prior = data_config.PRIOR_CLASS(**data_config.PRIOR_KWARGS)

    for physical_prior_cls in data_config.PHYSICAL_PRIOR_CLASSES:
        physical_prior = physical_prior_cls(**data_config.PRIOR_KWARGS)
        prior_ratio = PriorRatio(physical_prior, simulation_prior)
        get_ln_prior_ratio = np.vectorize(prior_ratio.ln_prior_ratio)

        for datadir in utils.TEST_DIR, utils.TRAINING_DIR:
            priordir = rundir/physical_prior_cls.__name__/datadir
            os.makedirs(priordir, exist_ok=True)
            filename = priordir/LN_PRIOR_RATIOS_FILENAME
            if not recompute_existing and filename.exists():
                print(f'Skipping existing {filename}...')
                continue

            parameters = pd.read_feather(
                rundir/datadir/utils.PARAMETERS_FILENAME
                )[simulation_prior.standard_params]

            ln_prior_ratios = get_ln_prior_ratio(**parameters)
            np.save(filename, ln_prior_ratios)


def _train_regressor_and_compute_weights(rundir,
                                         recompute_existing=False):
    rundir = Path(rundir)
    data_config = utils.load_data_config(rundir)
    prior_names = [prior_cls.__name__
                   for prior_cls in data_config.PHYSICAL_PRIOR_CLASSES]

    # Load train/test data
    traindir = rundir/utils.TRAINING_DIR
    mask_train = np.load(traindir/utils.MASK_FILENAME)
    compressed_data_train = np.load(
        traindir/utils.COMPRESSED_DATA_FILENAME)[mask_train]

    testdir = rundir/utils.TEST_DIR
    mask_test = np.load(testdir/utils.MASK_FILENAME)
    compressed_data_test = np.load(
        testdir/utils.COMPRESSED_DATA_FILENAME)[mask_test]

    for prior_name in prior_names:
        priordir = rundir/prior_name
        ln_prior_ratios_train = np.load(
            priordir/utils.TRAINING_DIR/LN_PRIOR_RATIOS_FILENAME)[mask_train]

        ln_prior_ratios_test = np.load(
            priordir/utils.TEST_DIR/LN_PRIOR_RATIOS_FILENAME)[mask_test]

        # Train (or load) regressor
        filename_mu = priordir/'ln-prior-ratio_regressor_mu.ubj'
        filename_sigma = priordir/'ln-prior-ratio_regressor_sigma.ubj'
        booster_mu = xgboost.XGBRegressor()
        booster_sigma = xgboost.XGBRegressor()
        if (not recompute_existing
                and filename_mu.exists()
                and filename_sigma.exists()):
            print(f'Loading existing {filename_mu} and {filename_sigma}...')
            booster_mu.load_model(filename_mu)
            booster_sigma.load_model(filename_sigma)
        else:
            booster_mu.fit(compressed_data_train, ln_prior_ratios_train)
            booster_mu.save_model(filename_mu)
            mu = booster_mu.predict(compressed_data_train)
            booster_sigma.fit(
                compressed_data_train,
                np.sqrt(np.abs(ln_prior_ratios_train**2 - mu**2)))
            booster_sigma.save_model(filename_sigma)

        # Compute weights
        _compute_and_save_weights(booster_mu,
                                  booster_sigma,
                                  compressed_data_test,
                                  ln_prior_ratios_test,
                                  priordir/utils.TEST_DIR,
                                  recompute_existing)

        _compute_and_save_weights(booster_mu,
                                  booster_sigma,
                                  compressed_data_train,
                                  ln_prior_ratios_train,
                                  priordir/utils.TRAINING_DIR,
                                  recompute_existing)


def _compute_and_save_weights(booster_mu, booster_sigma, compressed_data,
                              ln_prior_ratios, prior_datadir,
                              recompute_existing):
    filename = prior_datadir/utils.WEIGHTS_FILENAME
    if not recompute_existing and filename.exists():
        print(f'Skipping existing {filename}...')
        return

    mu = booster_mu.predict(compressed_data)
    sigma = booster_sigma.predict(compressed_data)

    result = differential_evolution(
        lambda x, *args: -_reweighting_efficiency(*x, *args),
        bounds=[(0, 2), (0, 4)],
        args=(mu, sigma, ln_prior_ratios))
    a, b = result.x
    ln_prior_ratios_pred = a * mu + b * sigma
    weights = np.exp(ln_prior_ratios - ln_prior_ratios_pred)
    print(f'Achieved reweighting efficiency of {-result.fun} '
          f'for {prior_datadir}.')
    np.save(filename, weights)


def _reweighting_efficiency(a, b, mu, sigma, ln_prior_ratios):
    ln_prior_ratios_pred = a * mu + b * sigma
    weights = np.exp(ln_prior_ratios - ln_prior_ratios_pred)
    return cogwheel.utils.n_effective(weights) / len(weights)


def main(rundir, recompute_existing=False):
    """
    Generate weights files in the training and test directories.

    There is a separate weights file for each physical prior in
    data_config.py, and for the training and test data.
    Each file contains importance sampling weights for each simulation.
    The weights are defined as

        (physical_prior / simulation_prior) / predicted_ratio

    where ``predicted_ratio`` is a function of the data only (the output
    of a regressor trained to predict physical_prior/simulation_prior,
    with a variance correction that is designed to maximize the
    reweighting efficiency).
    Weighting by the physical-to-simulation prior ratio ensures that the
    loss function is minimized when the network outputs the posterior
    under the *physical* prior. Dividing the weights by the predicted
    prior ratio mitigates the extra variance brought by importance
    sampling.

    Parameters
    ----------
    rundir : os.PathLike
        Run directory, must contain a training a test directories with
        simulation parameters (after ``generate_parameters.py`` has been
        run).

    recompute_existing : bool
        If a ln_prior_ratios file already exists, we only recompute it
        if this flag is True (default False).
    """
    _compute_ln_prior_ratios(rundir, recompute_existing)
    _train_regressor_and_compute_weights(rundir, recompute_existing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Compute weights to turn samples from a simulation prior
                       into a physical prior.''')
    parser.add_argument('rundir', help='path to a run directory.')
    parser.add_argument('--recompute_existing', action='store_true',
                        help='Recompute existing files.',)

    main(**vars(parser.parse_args()))
