"""
Rescale physical parameters with XGBoost models for the mean and scale
of the posterior.
"""

from pathlib import Path
import numpy as np

from xgboost import XGBRegressor

from cogwheel_machine import utils


MEAN_MODEL_FILENAME = 'model_mean.ubj'
SCALE_MODEL_FILENAME = 'model_mean_log_squared_err.ubj'


def rescale_parameters(rundir):
    """
    Fit mean and scale using XGBoost, and save rescaled parameters.

    This will create files for the XGBoost model of mean and
    mean-log-squared-error in `rundir`, and for the rescaled parameters
    in both the training and test directories.
    """
    rundir = Path(rundir)

    model_mean, model_mean_log_squared_err = _fit_xgboost(rundir)
    model_mean.save_model(rundir/MEAN_MODEL_FILENAME)
    model_mean_log_squared_err.save_model(rundir/SCALE_MODEL_FILENAME)


    for datadir in rundir/utils.TRAINING_DIR, rundir/utils.TEST_DIR:
        mask = np.load(datadir/utils.MASK_FILENAME)
        compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]

        folded_sampled_params = np.load(
            datadir/utils.FOLDED_SAMPLED_PARAMS_FILENAME)[mask]

        rescaled_parameters = rescale(folded_sampled_params, compressed_data,
                                      model_mean, model_mean_log_squared_err)
        np.save(datadir/utils.RESCALED_PARAMETERS_FILENAME,
                rescaled_parameters)


def load_models(rundir):
    """Load XGBoost models for the mean and mean-log-squared-error."""
    model_mean = XGBRegressor()
    model_mean.load_model(rundir/MEAN_MODEL_FILENAME)

    model_mean_log_squared_err = XGBRegressor()
    model_mean_log_squared_err.load_model(rundir/SCALE_MODEL_FILENAME)

    return model_mean, model_mean_log_squared_err


def rescale(folded_sampled_params, compressed_data, model_mean,
            model_mean_log_squared_err):
    """Apply rescaling to physical parameters to make them ~N(0, 1)."""
    sigma_pred = _get_sigma(model_mean_log_squared_err, compressed_data)

    mean_pred = model_mean.predict(compressed_data)
    rescaled_parameters = (folded_sampled_params - mean_pred) / sigma_pred
    return rescaled_parameters


def unrescale(rescaled_parameters, compressed_data, model_mean,
              model_mean_log_squared_err):
    """Inverse of ``rescale``."""
    sigma_pred = _get_sigma(model_mean_log_squared_err, compressed_data)

    mean_pred = model_mean.predict(compressed_data)
    folded_sampled_params = rescaled_parameters * sigma_pred + mean_pred
    return folded_sampled_params


def _get_sigma(model_mean_log_squared_err, compressed_data):
    # ``factor`` turns the exp(mean-log-squared-error / 2) into std:
    # 1/exp(1/2 integral_(-∞)^∞ (exp(-x^2/2) log(x^2))/sqrt(2 π) dx)
    # = e^(1/2 ( gamma + log(2))) ≈ 1.88736
    factor = np.exp(np.euler_gamma / 2) * np.sqrt(2)
    return factor * np.exp(
        model_mean_log_squared_err.predict(compressed_data) / 2)


def _fit_xgboost(rundir):
    rundir = Path(rundir)
    datadir = rundir/utils.TRAINING_DIR
    config = utils.load_data_config(rundir)

    mask = np.load(datadir/utils.MASK_FILENAME)
    compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]

    folded_sampled_params = np.load(
        datadir/utils.FOLDED_SAMPLED_PARAMS_FILENAME)[mask]

    model_mean = XGBRegressor(**config.XGBOOST_KWARGS)
    model_mean.fit(compressed_data, folded_sampled_params)

    log_squared_err = np.log(
        (folded_sampled_params - model_mean.predict(compressed_data))**2
        + 1e-10)  # Prevent log(0) if the model gets it perfect

    model_mean_log_squared_err = XGBRegressor(**config.XGBOOST_KWARGS)
    model_mean_log_squared_err.fit(compressed_data, log_squared_err)
    return model_mean, model_mean_log_squared_err
