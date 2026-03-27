"""Compute weights to go from a simulation prior to a physical prior."""
import argparse
import functools
import json
import os
from pathlib import Path

import xgboost
import numpy as np
import pandas as pd
import cogwheel.utils

from scipy.optimize import differential_evolution

from cogwheel.prior_ratio import PriorRatio

from . import condor_utils, utils


LN_PRIOR_RATIOS_FILENAME = 'ln_prior_ratios.npy'

_REGRESSOR_MU_FILENAME = 'ln-prior-ratio_regressor_mu.ubj'
_REGRESSOR_SIGMA_FILENAME = 'ln-prior-ratio_regressor_sigma.ubj'
_COEFFICIENTS_FILENAME = 'coefficients.json'


def _compute_ln_prior_ratios(priordir, recompute_existing=False):
    """
    Generate files with log prior ratios in the training and test
    directories.

    The files contain a log(physical_prior / simulation_prior) for each
    simulation. There is a separate directory and file for the training
    and test data.

    Parameters
    ----------
    priordir : pathlib.Path
        Directory inside ``rundir``, containing a ``prior_config.py``
        file. Output of :py:func:`utils.setup_priordir`.

    recompute_existing : bool
        If a ln_prior_ratios.npy file already exists, we only recompute
        it if this flag is True (default False).
    """
    rundir = priordir.parent

    data_config = utils.load_data_config(rundir)
    simulation_prior = data_config.PRIOR_CLASS(**data_config.PRIOR_KWARGS)

    prior_config = utils.load_prior_config(priordir)
    prior_kwargs = data_config.PRIOR_KWARGS | prior_config.PRIOR_KWARGS
    physical_prior = prior_config.PRIOR_CLASS(**prior_kwargs)

    prior_ratio = PriorRatio(physical_prior, simulation_prior)
    get_ln_prior_ratio = np.vectorize(prior_ratio.ln_prior_ratio)

    for datadir in utils.TEST_DIR, utils.TRAINING_DIR:
        os.makedirs(priordir/datadir, exist_ok=True)
        filename = priordir/datadir/LN_PRIOR_RATIOS_FILENAME
        if not recompute_existing and filename.exists():
            print(f'Skipping existing {filename}...')
            continue

        parameters = pd.read_feather(
            rundir/datadir/utils.PARAMETERS_FILENAME
            )[simulation_prior.standard_params]

        ln_prior_ratios = get_ln_prior_ratio(**parameters)
        np.save(filename, ln_prior_ratios)


def _train_regressor(priordir, recompute_existing=False):
    filename_mu = priordir/_REGRESSOR_MU_FILENAME
    filename_sigma = priordir/_REGRESSOR_SIGMA_FILENAME
    filename_coefs = priordir/_COEFFICIENTS_FILENAME

    if (not recompute_existing
            and filename_mu.exists()
            and filename_sigma.exists()
            and filename_coefs.exists()):
        print('Previous counterweight regressor exists, skip training...')
        return

    # Load training data
    compressed_data_train, ln_prior_ratios_train \
        = _load_compressed_data_and_ln_prior_ratios(
            priordir, utils.TRAINING_DIR)

    # Train regressor for mu and sigma, fit their coefficients

    # Note: weights are of the form
    # ln_counterweights = coef_mu * mu + coef_sigma * sigma
    # ln_weights = ln_prior_ratios - ln_counterweights

    ## mu
    booster_mu = xgboost.XGBRegressor()
    booster_mu.fit(compressed_data_train, ln_prior_ratios_train)
    booster_mu.save_model(filename_mu)
    mu_train = booster_mu.predict(compressed_data_train)

    ## sigma
    booster_sigma = xgboost.XGBRegressor()
    booster_sigma.fit(
        compressed_data_train,
        np.sqrt(np.abs(ln_prior_ratios_train**2 - mu_train**2)))
    booster_sigma.save_model(filename_sigma)
    sigma_train = booster_sigma.predict(compressed_data_train)

    ## coefs

    def neg_efficiency(coefs):
        """-n_eff / N, function to minimize."""
        weights = _weights(ln_prior_ratios_train, mu_train, sigma_train,
                           *coefs)
        return - cogwheel.utils.n_effective(weights) / len(weights)

    result = differential_evolution(neg_efficiency, bounds=[(0, 2), (0, 4)])
    coefs = {'coef_mu': result.x[0],
             'coef_sigma': result.x[1]}
    with open(filename_coefs, 'w', encoding='utf-8') as file_coefs:
        json.dump(coefs, file_coefs, indent=2)


def _compute_weights(priordir, dataset_type, recompute_existing=False):
    filename = priordir/dataset_type/utils.WEIGHTS_FILENAME
    if not recompute_existing and filename.exists():
        print(f'Skipping existing {filename}...')
        return

    booster_mu = xgboost.XGBRegressor()
    booster_mu.load_model(priordir/_REGRESSOR_MU_FILENAME)

    booster_sigma = xgboost.XGBRegressor()
    booster_sigma.load_model(priordir/_REGRESSOR_SIGMA_FILENAME)

    with open(priordir/_COEFFICIENTS_FILENAME, encoding='utf-8') as file_coefs:
        coefs = json.load(file_coefs)

    compressed_data, ln_prior_ratios \
        = _load_compressed_data_and_ln_prior_ratios(priordir, dataset_type)

    mu = booster_mu.predict(compressed_data)
    sigma = booster_sigma.predict(compressed_data)
    weights = _weights(ln_prior_ratios, mu, sigma, **coefs)

    eff = cogwheel.utils.n_effective(weights) / len(weights)
    print(f'Achieved reweighting efficiency of {eff:.4g} for {filename}.')

    np.save(filename, weights)


@functools.lru_cache(maxsize=1)
def _load_compressed_data_and_ln_prior_ratios(priordir, dataset_type):
    rundir = priordir.parent
    datadir = rundir/dataset_type
    mask = np.load(datadir/utils.MASK_FILENAME)
    compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]
    ln_prior_ratios = np.load(priordir/dataset_type/LN_PRIOR_RATIOS_FILENAME
                             )[mask]
    return compressed_data, ln_prior_ratios


def _weights(ln_prior_ratios, mu, sigma, coef_mu, coef_sigma):
    ln_counterweights = coef_mu * mu + coef_sigma * sigma
    return np.exp(ln_prior_ratios - ln_counterweights)


def setup_condor_sub(priordir, request_memory='16G', request_disk='1G',
                     submit=False, **submit_kwargs):
    """
    Create a script to run the weighting job on HTCondor.

    This will generate the following files:
        {priordir}/submission_scripts/weighting.{sub,sh}

    Parameters
    ----------
    priordir : os.PathLike
        Directory inside ``rundir``, containing a ``prior_config.py``
        file. Output of :py:func:`utils.setup_priordir`.

    Returns
    -------
    pathlib.Path
        Path to the HTCondor submit file.

    See Also
    --------
    utils.setup_priordir
    """
    priordir = Path(priordir).resolve()
    stem = priordir/'submission_scripts'/'weighting'
    module = 'labrador.weighting'
    return condor_utils.setup_condor_sub(stem, module,
                                         request_memory=request_memory,
                                         request_disk=request_disk,
                                         submit=submit,
                                         arguments=priordir,
                                         **submit_kwargs)


def main(priordir, recompute_existing=False):
    """
    Generate weights files in the training and test directories.

    There is a separate directory for the training and test set inside
    priordir with a weights file.
    Each file contains importance sampling weights for each simulation.
    The weights are defined as::

        ln_weights = ln_prior_ratios - ln_counterweights
        ln_counterweights = coef_mu * mu + coef_sigma * sigma

    where ``ln_counterweights`` is a function of the data only (the
    output of a regressor trained to predict
    physical_prior/simulation_prior, with a variance correction that is
    designed to maximize the reweighting efficiency).
    Weighting by the physical-to-simulation prior ratio ensures that the
    loss function is minimized when the network outputs the posterior
    under the *physical* prior. Applying the counterweights mitigates
    the extra variance brought by importance sampling.

    Parameters
    ----------
    priordir : os.PathLike
        Directory inside ``rundir``, containing a ``prior_config.py``
        file. Output of :py:func:`utils.setup_priordir`.

    recompute_existing : bool
        If a ln_prior_ratios, regressor or weights file already exists,
        we only recompute it if this flag is True (default False).

    See Also
    --------
    utils.setup_priordir
    """
    priordir = Path(priordir).resolve()
    _compute_ln_prior_ratios(priordir, recompute_existing)
    _train_regressor(priordir, recompute_existing)
    _compute_weights(priordir, utils.TRAINING_DIR)
    _compute_weights(priordir, utils.TEST_DIR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Compute weights to turn samples from a simulation prior
                       into a physical prior.''')
    parser.add_argument('priordir', help='path to a prior-directory.')
    parser.add_argument('--recompute_existing', action='store_true',
                        help='Recompute existing files.',)

    main(**vars(parser.parse_args()))
