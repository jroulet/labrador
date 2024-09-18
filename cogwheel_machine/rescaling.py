"""
Rescale physical parameters with XGBoost models for the mean and scale
of the posterior.
"""

from pathlib import Path
import numpy as np
import pandas as pd

import xgboost

import cogwheel.utils

from cogwheel_machine import utils


MEAN_MODEL_FILENAME = 'model_mean.ubj'
SCALE_MODEL_FILENAME = 'model_mean_log_squared_err.ubj'


def rescale_parameters(rundir):
    """
    Fit mean and scale using XGBoost, and save rescaled parameters.

    This will create files for the XGBoost model of mean and
    mean-log-squared-error in `rundir` (if not already present), and for
    the rescaled parameters in both the training and test directories.
    """
    rundir = Path(rundir)

    parameter_rescaler = ParameterRescaler(rundir)

    for datadir in rundir/utils.TRAINING_DIR, rundir/utils.TEST_DIR:
        mask = np.load(datadir/utils.MASK_FILENAME)
        compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]

        folded_sampled_params = np.load(
            datadir/utils.FOLDED_SAMPLED_PARAMS_FILENAME)[mask]

        rescaled_parameters = parameter_rescaler.rescale(compressed_data,
                                                         folded_sampled_params)

        np.save(datadir/utils.RESCALED_PARAMETERS_FILENAME,
                rescaled_parameters)


class ParameterRescaler:
    """
    Rescale physical parameters to make them ~N(0, 1), or unrescale them
    back.

    Bounded parameters are mapped to (-inf, inf) to avoid hard edges.
    Periodic parameters are first centered by predicting their circular
    mean from the data (so as to avoid spurious multimodality when the
    posterior straddles the branch cut) and then mapped to (-inf, inf).

    The top-level function ``rescale_parameters`` provides an interface
    for this class that is suitable for simple use cases.

    Methods
    -------
    .rescale
    .unrescale
    """

    def __init__(self, rundir):
        self.rundir = Path(rundir)
        self.config = utils.load_data_config(self.rundir)

        self.folded_range_dic = self._get_folded_range_dic()
        self.bounded_params = self._get_bounded_params()

        assert set(self.bounded_params) <= self.folded_range_dic.keys()
        assert set(self.periodic_params) <= self.folded_range_dic.keys()

        self.model_mean = None  # Set by ._{load|fit}_models
        self.model_mean_log_squared_err = None  # Set by ._{load|fit}_models
        try:
            self._load_models()
        except xgboost.core.XGBoostError:  # Models have not been trained yet
            self._fit_models()
            self.model_mean.save_model(self.rundir/MEAN_MODEL_FILENAME)
            self.model_mean_log_squared_err.save_model(
                self.rundir/SCALE_MODEL_FILENAME)

    @property
    def periodic_params(self):
        """List of periodic-parameter names."""
        return self.config.TRANSFORM_CLASS.periodic_params

    @property
    def bounded_nonperiodic_params(self):
        """List of bounded-non-periodic-parameter names."""
        return [par for par in self.bounded_params
                if par not in self.periodic_params]

    def rescale(self, compressed_data, folded_sampled_params):
        """
        Apply rescaling to physical parameters to make them ~N(0, 1).
        """
        # TODO: change other code so folded_sampled_params is a DataFrame
        parameters = pd.DataFrame(folded_sampled_params,
                                  columns=self.folded_range_dic.keys())
        self._decompactify_bounded_nonperiodic(parameters)
        self._periodic_to_angle(parameters)
        self._remove_mean(compressed_data, parameters)
        self._decompactify_periodic(parameters)
        self._remove_scale(compressed_data, parameters)

        # Ensure parameters are returned in the correct order
        return parameters[list(self.folded_range_dic)].to_numpy()

    def unrescale(self, compressed_data, rescaled_params):
        """
        Map rescaled parameters from ~N(0, 1) to their physical range.

        Inverse of ``.rescale``.
        """
        parameters = pd.DataFrame(rescaled_params,
                                  columns=self.folded_range_dic.keys())
        self._add_scale(compressed_data, parameters)
        self._compactify_periodic(parameters)
        self._add_mean(compressed_data, parameters)
        self._angle_to_periodic(parameters)
        self._compactify_bounded_nonperiodic(parameters)

        # Ensure parameters are returned in the correct order
        return parameters[list(self.folded_range_dic)]

    def _decompactify_bounded_nonperiodic(self,
                                          parameters: pd.DataFrame):
        """
        Decompactify columns for ``.bounded_nonperiodic_params``
        inplace.
        """
        for par in self.bounded_nonperiodic_params:
            parameters[par] = _decompactify(parameters[par],
                                            *self.folded_range_dic[par])

    def _compactify_bounded_nonperiodic(self, parameters: pd.DataFrame):
        """
        Compactify columns for ``.bounded_nonperiodic_params``
        inplace.
        """
        for par in self.bounded_nonperiodic_params:
            parameters[par] = _compactify(parameters[par],
                                          *self.folded_range_dic[par])

    def _periodic_to_angle(self, parameters: pd.DataFrame):
        """Map the periodic parameters to (-pi, pi) inplace."""
        for par in self.periodic_params:
            parameters[par] = np.interp(parameters[par],
                                        self.folded_range_dic[par],
                                        (-np.pi, np.pi))

    def _angle_to_periodic(self, parameters: pd.DataFrame):
        """
        Map the periodic parameters from (-pi, pi) to their physical
        range inplace.
        Inverse of ``._periodic_to_angle``.
        """
        for par in self.periodic_params:
            parameters[par] = np.interp(parameters[par],
                                        (-np.pi, np.pi),
                                        self.folded_range_dic[par])

    def _remove_mean(self, compressed_data, parameters: pd.DataFrame):
        """
        Remove mean of the parameters inplace, using circular mean for
        the periodic ones.

        Note: by now bounded parameters should have been decompactified,
        and periodic parameters have been mapped to angles.
        """
        parameters -= self._get_mean(compressed_data)

        for par in self.periodic_params:
            parameters[par] = cogwheel.utils.mod(parameters[par], start=-np.pi)

    def _add_mean(self, compressed_data, parameters: pd.DataFrame):
        """
        Add mean of the parameters back inplace, using circular mean for
        the periodic ones.
        Inverse of ``._remove_mean``
        """
        parameters += self._get_mean(compressed_data)

        for par in self.periodic_params:
            parameters[par] = cogwheel.utils.mod(parameters[par], start=-np.pi)

    def _get_mean(self, compressed_data) -> pd.DataFrame:
        """
        Predict mean of the parameters, using circular mean for the
        periodic ones.
        """
        mean = self._series_or_dataframe(self.model_mean.predict(compressed_data),
                                         labels=self._model_mean_params)

        # Compute circular mean of periodic parameters
        for par in self.periodic_params:
            mean[par] = np.arctan2(mean.pop(f'_sin_{par}'),
                                   mean.pop(f'_cos_{par}'))

        return mean

    def _decompactify_periodic(self, parameters: pd.DataFrame):
        """
        Decompactify columns for ``.periodic_params`` inplace.

        By now the parameters should have been turned into angles in
        (-pi, pi), and had their circular mean subtracted.
        """
        for par in self.periodic_params:
            parameters[par] = _decompactify(parameters[par], -np.pi, np.pi)

    def _compactify_periodic(self, parameters: pd.DataFrame):
        """
        Compactify columns for ``.periodic_params`` inplace.
        Inverse of ``._decompactify_periodic``.
        """
        for par in self.periodic_params:
            parameters[par] = _compactify(parameters[par], -np.pi, np.pi)

    def _remove_scale(self, compressed_data, parameters: pd.DataFrame):
        """Divide parameters by their predicted scale inplace."""
        parameters /= self._get_scale(compressed_data)

    def _add_scale(self, compressed_data, parameters: pd.DataFrame):
        """Multiply parameters by their predicted scale inplace."""
        parameters *= self._get_scale(compressed_data)

    def _get_scale(self, compressed_data) -> pd.DataFrame:
        """
        Return prediction for the standard deviation of each parameter
        (after decompactifying bounded parameters).
        """
        # ``factor`` turns the exp(mean-log-squared-error / 2) into std:
        # 1/exp(1/2 integral_(-∞)^∞ (exp(-x^2/2) log(x^2))/sqrt(2 π) dx)
        # = e^(1/2 (gamma + log(2))) ≈ 1.88736
        factor = np.exp(np.euler_gamma / 2) * np.sqrt(2)
        mean_log_squared_err = self.model_mean_log_squared_err.predict(
            compressed_data)

        return self._series_or_dataframe(factor * np.exp(mean_log_squared_err / 2),
                                         labels=list(self.folded_range_dic))

    def _load_models(self):
        """
        Load XGBoost models for the mean and mean-log-squared-error.
        If any of the files does not exist, raise
        ``xgboost.core.XGBoostError``.
        """
        model_mean = xgboost.XGBRegressor()
        model_mean.load_model(self.rundir/MEAN_MODEL_FILENAME)

        model_mean_log_squared_err = xgboost.XGBRegressor()
        model_mean_log_squared_err.load_model(self.rundir/SCALE_MODEL_FILENAME)

        self.model_mean = model_mean
        self.model_mean_log_squared_err = model_mean_log_squared_err

    def _fit_models(self):
        """
        Set attributes ``.model_mean`` and
        ``.model_mean_log_squared_err`` by training XGBoost.
        """
        datadir = self.rundir/utils.TRAINING_DIR

        mask = np.load(datadir/utils.MASK_FILENAME)
        parameters = pd.DataFrame(
            np.load(datadir/utils.FOLDED_SAMPLED_PARAMS_FILENAME)[mask],
            columns=self.folded_range_dic.keys())

        compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]

        self._decompactify_bounded_nonperiodic(parameters)
        self._periodic_to_angle(parameters)
        self._fit_mean(compressed_data, parameters)
        self._remove_mean(compressed_data, parameters)
        self._decompactify_periodic(parameters)
        self._fit_scale(compressed_data, parameters)

    def _fit_mean(self, compressed_data, parameters):
        """
        Fit the mean of the parameters as a function of data with
        XGBoost and set the ``.model_mean`` attribute.
        """
        model_mean_parameters = parameters[self._nonperiodic_params]
        for par in self.periodic_params:
            model_mean_parameters[f'_sin_{par}'] = np.sin(parameters[par])
            model_mean_parameters[f'_cos_{par}'] = np.cos(parameters[par])

        assert list(model_mean_parameters) == self._model_mean_params
        model_mean = xgboost.XGBRegressor(**self.config.XGBOOST_KWARGS)
        print('Training XGBoost model for the posterior mean...')
        model_mean.fit(compressed_data, model_mean_parameters)
        self.model_mean = model_mean

    def _fit_scale(self, compressed_data, parameters):
        """
        Fit the mean-log-squared-error of the parameters as a function
        of data with XGBoost and set the ``.model_mean_log_squared_err``
        attribute.
        """
        # Prevent log(0) if the ``model_mean`` got it perfect:
        log_squared_err = np.log(parameters**2 + 1e-10)

        model_mean_log_squared_err = xgboost.XGBRegressor(
            **self.config.XGBOOST_KWARGS)
        print('Training XGBoost model for the posterior scale...')
        model_mean_log_squared_err.fit(compressed_data, log_squared_err)

        self.model_mean_log_squared_err = model_mean_log_squared_err

    @staticmethod
    def _series_or_dataframe(arr, labels):
        # If `arr` has a single row, return a Series so it broadcasts
        n_rows, _ = arr.shape
        if n_rows == 1:
            return pd.Series(arr[0], labels)
        return pd.DataFrame(arr, columns=labels)

    def _get_folded_range_dic(self):
        """
        Return the range_dic of the transform class, setting the value for
        ``'lnq'`` from the config.
        """
        # Somewhat fragile, but these methods could be overriden if needed
        folded_range_dic = self.config.TRANSFORM_CLASS.range_dic.copy()
        if 'lnq' in folded_range_dic:
            folded_range_dic['lnq'] = (
                np.log(self.config.PRIOR_KWARGS['q_min']), 0.0)

        for par in self.config.TRANSFORM_CLASS.folded_params:
            # Divide range of folded parameters in two
            folded_range_dic[par] = (folded_range_dic[par][0],
                                     np.mean(folded_range_dic[par]))

        return folded_range_dic

    def _get_bounded_params(self):
        """Return list of parameters that have a finite range."""
        return [par for par, (low, high) in self.folded_range_dic.items()
                if np.isfinite(high - low)]

    @property
    def _model_mean_params(self):
        """Parameters predicted by the XGBoost model for the mean."""
        parameters = list(self.folded_range_dic)
        for par in self.periodic_params:
            parameters.remove(par)
            parameters.extend([f'_sin_{par}', f'_cos_{par}'])
        return parameters

    @property
    def _nonperiodic_params(self):
        """Parameters predicted by the XGBoost model for the mean."""
        return [par for par in self.folded_range_dic
                if par not in self.periodic_params]


def _compactify(value, a, b):
    """
    Compactify a value from an infinite interval to a finite interval
    [a, b] using tanh.

    Parameters
    ----------
    value: float
        The value to be compactified.

    a: float
        The lower bound of the finite interval.

    b: float
        The upper bound of the finite interval.

    Returns
    -------
    float: The compactified value within the interval [a, b].
    """
    return (b - a) / 2 * np.tanh(value) + (b + a) / 2


def _decompactify(compact_value, a, b):
    """
    Decompactify a value from a finite interval [a, b] to an infinite
    interval using arctanh.

    Parameters
    ----------
    compact_value: float
        The compactified value within the interval [a, b].

    a: float
        The lower bound of the finite interval.

    b: float
        The upper bound of the finite interval.

    Returns
    -------
    float: The decompactified value within the infinite interval.
    """
    return np.arctanh(2 * (compact_value - (b + a) / 2) / (b - a))
