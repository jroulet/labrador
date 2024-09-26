"""
Rescale physical parameters with XGBoost models for the mean and scale
of the posterior.
"""
import argparse
from pathlib import Path
import numpy as np

import torch
from torch import nn

import cogwheel.utils

from cogwheel_machine import utils, sbi_hacks


PARAMETER_RESCALER_TRAINING_FILENAME = 'parameter_rescaler_training.pth'
PARAMETER_RESCALER_FILENAME = 'parameter_rescaler.pth'


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

        self.device = torch.device(
            self.config.DEVICE
            or ('cuda' if torch.cuda.is_available() else 'cpu'))

        params = list(self.folded_range_dic)
        self._periodic_inds = [
            params.index(par) for par in self.periodic_params]

        self._bounded_nonperiodic_inds = [
            params.index(par) for par in self.bounded_nonperiodic_params]

        self._nonperiodic_inds = [ind for ind in range(self.n_parameters)
                                  if ind not in self._periodic_inds]

        self._nonperiodic_mean = None  # Set by ._{load|fit}_model
        self._nonperiodic_scale = None  # Set by ._{load|fit}_model
        self._moments_model = None  # Set by ._{load|fit}_model
        self._training_info = None  # Set by ._{load|fit}_model
        try:
            self._load_model()
        except FileNotFoundError:  # Models have not been trained yet
            self._setup_model()
            self.train()

    @property
    def n_parameters(self):
        return len(self.folded_range_dic)

    @property
    def periodic_params(self):
        """List of periodic-parameter names."""
        return self.config.TRANSFORM_CLASS.periodic_params

    @property
    def bounded_nonperiodic_params(self):
        """List of bounded-non-periodic-parameter names."""
        return [par for par in self.bounded_params
                if par not in self.periodic_params]

    def rescale(self, compressed_data, folded_sampled_params,
                double_precision=True):
        """
        Apply rescaling to physical parameters to make them ~N(0, 1).
        """
        compressed_data = torch.as_tensor(compressed_data).to(self.device)
        parameters = torch.as_tensor(folded_sampled_params).to(self.device)
        if double_precision:
            parameters = parameters.double()

        mean, chol_inv = self._predict_moments(compressed_data)
        return self._rescale(parameters, mean, chol_inv)

    def _rescale(self, parameters, mean, chol_inv):
        parameters = parameters.clone().detach()
        self._decompactify_bounded_nonperiodic(parameters)
        self._standardize_nonperiodic(parameters)
        self._periodic_to_angle(parameters)
        self._remove_mean(mean, parameters)
        self._decompactify_periodic(parameters)
        parameters = self._remove_scale(chol_inv, parameters)
        return parameters

    def unrescale(self, compressed_data, rescaled_params,
                  double_precision=True):
        """
        Map rescaled parameters from ~N(0, 1) to their physical range.

        Inverse of ``.rescale``.
        """
        compressed_data = torch.as_tensor(compressed_data).to(self.device)
        parameters = torch.as_tensor(rescaled_params).to(self.device)
        if double_precision:
            parameters = parameters.double()

        mean, chol_inv = self._predict_moments(compressed_data)
        return self._unrescale(parameters, mean, chol_inv)

    def _unrescale(self, parameters, mean, chol_inv):
        parameters = parameters.clone().detach()
        parameters = self._add_scale(chol_inv, parameters)
        self._compactify_periodic(parameters)
        self._add_mean(mean, parameters)
        self._angle_to_periodic(parameters)
        self._unstandardize_nonperiodic(parameters)
        self._compactify_bounded_nonperiodic(parameters)

        return parameters

    def _decompactify_bounded_nonperiodic(self, parameters):
        """
        Decompactify columns for ``.bounded_nonperiodic_params``
        inplace.
        """
        for i, par in zip(self._bounded_nonperiodic_inds,
                          self.bounded_nonperiodic_params):
            parameters[..., i] = _decompactify(parameters[..., i],
                                               *self.folded_range_dic[par])

    def _compactify_bounded_nonperiodic(self, parameters):
        """
        Compactify columns for ``.bounded_nonperiodic_params`` inplace.
        """
        for i, par in zip(self._bounded_nonperiodic_inds,
                          self.bounded_nonperiodic_params):
            parameters[..., i] = _compactify(parameters[..., i],
                                             *self.folded_range_dic[par])

    def _standardize_nonperiodic(self, parameters):
        parameters[..., self._nonperiodic_inds] -= self._nonperiodic_mean
        parameters[..., self._nonperiodic_inds] /= self._nonperiodic_scale

    def _unstandardize_nonperiodic(self, parameters):
        parameters[..., self._nonperiodic_inds] *= self._nonperiodic_scale
        parameters[..., self._nonperiodic_inds] += self._nonperiodic_mean

    def _periodic_to_angle(self, parameters):
        """Map the periodic parameters to (-pi, pi) inplace."""
        for i, par in zip(self._periodic_inds, self.periodic_params):
            parameters[..., i] = self._linear_rescale(
                parameters[..., i],
                self.folded_range_dic[par],
                (-np.pi, np.pi))

    def _angle_to_periodic(self, parameters):
        """
        Map the periodic parameters from (-pi, pi) to their physical
        range inplace.
        Inverse of ``._periodic_to_angle``.
        """
        for i, par in zip(self._periodic_inds, self.periodic_params):
            parameters[..., i] = self._linear_rescale(
                parameters[..., i],
                (-np.pi, np.pi),
                self.folded_range_dic[par])

    @staticmethod
    def _linear_rescale(x, x_rng, y_rng):
        slope = (y_rng[1] - y_rng[0]) / (x_rng[1] - x_rng[0])
        return y_rng[0] + (x - x_rng[0]) * slope

    def _remove_mean(self, mean, parameters):
        """
        Remove mean of the parameters, using circular mean for
        the periodic ones.

        Note: by now bounded parameters should have been decompactified,
        and periodic parameters have been mapped to angles.
        """
        parameters -= mean
        self._mod_periodic(parameters)

    def _add_mean(self, mean, parameters):
        """
        Add mean of the parameters back inplace, using circular mean for
        the periodic ones.
        Inverse of ``._remove_mean``
        """
        parameters += mean
        self._mod_periodic(parameters)

    def _mod_periodic(self, parameters):
        """
        Map periodic parameters to (-pi, pi) inplace by adding a
        multiple of 2 pi.
        """
        for i in self._periodic_inds:
            parameters[..., i] = cogwheel.utils.mod(parameters[..., i],
                                                    start=-np.pi)

    def _decompactify_periodic(self, parameters):
        """
        Decompactify columns for ``.periodic_params`` inplace.

        By now the parameters should have been turned into angles in
        (-pi, pi), and had their circular mean subtracted.
        """
        for i in self._periodic_inds:
            parameters[..., i] = _decompactify(parameters[..., i],
                                               -np.pi, np.pi)

    def _compactify_periodic(self, parameters):
        """
        Compactify columns for ``.periodic_params`` inplace.
        Inverse of ``._decompactify_periodic``.
        """
        for i in self._periodic_inds:
            parameters[..., i] = _compactify(parameters[..., i], -np.pi, np.pi)

    def _remove_scale(self, chol_inv, parameters):
        """Divide parameters by their predicted scale."""
        chol_inv = chol_inv.to(parameters.dtype)
        return torch.einsum('...j,...jk', parameters, chol_inv)

    def _add_scale(self, chol_inv, parameters):
        """Multiply parameters by their predicted scale."""
        chol_inv = chol_inv.to(parameters.dtype)
        return torch.linalg.solve_triangular(
            chol_inv, parameters.unsqueeze(-2), upper=False, left=False
            ).squeeze(-2)

    def _load_model(self):
        """
        Set attributes ``_nonperiodic_mean``, ``_nonperiodic_scale`` and
        ``_moments_model`` by loading from disk.
        """
        model_config = torch.load(self.rundir/PARAMETER_RESCALER_FILENAME,
                                  weights_only=True)

        self._nonperiodic_mean = model_config['nonperiodic_mean']
        self._nonperiodic_scale = model_config['nonperiodic_scale']

        self._moments_model = _MultiLayerPerceptron.from_dict(
            model_config['_MultiLayerPerceptron']).to(self.device)

        self._training_info = torch.load(
            self.rundir/PARAMETER_RESCALER_TRAINING_FILENAME,
            weights_only=True)

    def _save_current_model(self):
        model_config = {'_MultiLayerPerceptron': self._moments_model.to_dict(),
                        'nonperiodic_mean': self._nonperiodic_mean,
                        'nonperiodic_scale': self._nonperiodic_scale}

        torch.save(model_config, self.rundir/PARAMETER_RESCALER_FILENAME)

    def _setup_model(self):
        """
        Set attributes ``_non_periodic_mean`` and ``non_periodic_scale``
        by measuring them from the dataset, and ``_moments_model`` by
        training a neural network.
        """
        compressed_data, parameters, _, _ = self._load_data()

        n_inputs = compressed_data.shape[1]

        # Mean and Cholesky decomposition of the inverse covariance;
        # periodic parameters require an extra output because we predict
        # the mean sin and cos.
        n_outputs = (self.n_parameters + len(self._periodic_inds)
                     + self.n_parameters * (self.n_parameters + 1) // 2)
        self._moments_model = _MultiLayerPerceptron(
            n_inputs, n_outputs, **self.config.RESCALER_NN_KWARGS
            ).to(self.device)

        self._fit_global_moments(parameters)

        # TODO could add optimizer state dict
        self._training_info = {'train_losses': [],
                               'val_losses': [],
                               'best_val_loss': np.inf}

    def train(self):
        kwargs = self.config.RESCALER_TRAIN_KWARGS
        optimizer = torch.optim.Adam(self._moments_model.parameters(),
                                     **kwargs['optimizer_kwargs'])

        train_loader, val_loader = self._get_dataloaders()

        for _ in range(kwargs['max_num_epochs']):
            self._moments_model.train()

            train_loss = 0
            for training_batch in train_loader:
                optimizer.zero_grad()
                loss = self._loss_function(*training_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            self._training_info['train_losses'].append(train_loss)

            self._moments_model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    loss = self._loss_function(*val_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self._training_info['val_losses'].append(val_loss)

            if val_loss < self._training_info['best_val_loss']:
                self._training_info['best_val_loss'] = val_loss
                patience_counter = 0
                self._save_current_model()
            else:  # Early stopping
                patience_counter += 1
                if patience_counter >= kwargs['stop_after_epochs']:
                    break

        self._save_training_info()
        self._load_model()

    def _save_training_info(self):
        torch.save(self._training_info,
                   self.rundir/PARAMETER_RESCALER_TRAINING_FILENAME)

    def _get_dataloaders(self):
        kwargs = self.config.RESCALER_TRAIN_KWARGS

        dataset = torch.utils.data.TensorDataset(*self._load_data())

        train_ind_batches, val_ind_batches \
            = sbi_hacks.get_train_val_batch_inds(
                len(dataset),
                kwargs['training_batch_size'],
                kwargs['validation_fraction'])

        train_batches = [dataset[inds] for inds in train_ind_batches]
        val_batches = [dataset[inds] for inds in val_ind_batches]

        train_loader = sbi_hacks.FixedBatchesDataLoader(train_batches)
        val_loader = sbi_hacks.FixedBatchesDataLoader(val_batches)

        return train_loader, val_loader

    def _load_data(self):
        datadir = self.rundir/utils.TRAINING_DIR

        mask = np.load(datadir/utils.MASK_FILENAME)
        compressed_data = torch.from_numpy(
            np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]
            ).to(self.device)
        parameters = torch.from_numpy(
            np.load(datadir/utils.FOLDED_SAMPLED_PARAMS_FILENAME)[mask]
            ).to(self.device)

        return compressed_data, parameters, sin, cos

    def _get_sin_cos_periodic_parameters(self, parameters):
        parameters = parameters.clone().detach()
        self._periodic_to_angle(parameters)
        angle = parameters[:, self._periodic_inds]
        return torch.sin(angle), torch.cos(angle)

    def _fit_global_moments(self, parameters):
        parameters = parameters.clone().detach()
        self._decompactify_bounded_nonperiodic(parameters)
        nonperiodic = parameters[:, self._nonperiodic_inds]
        self._nonperiodic_mean = torch.mean(nonperiodic, dim=0)
        self._nonperiodic_scale = torch.std(nonperiodic, dim=0)

    def _predict_moments(self, compressed_data, ret_mean_sin_cos=False):
        output = self._moments_model(compressed_data)

        split_sizes = (len(self._nonperiodic_inds),
                       len(self._periodic_inds),
                       len(self._periodic_inds),
                       self.n_parameters * (self.n_parameters + 1) // 2)
        *means, chol_inv_values = torch.split(output, split_sizes, dim=-1)

        mean = self._get_mean(*means)
        chol_inv = self._get_cholesky(chol_inv_values)

        if ret_mean_sin_cos:
            return mean, chol_inv, *means[1:]
        return mean, chol_inv

    def _get_mean(self, mean_nonperiodic, mean_sin_periodic,
                  mean_cos_periodic):
        """
        Predict mean of the parameters, using circular mean for the
        periodic ones.
        """
        *pre_shape, _ = mean_nonperiodic.shape
        mean = torch.empty(*pre_shape, self.n_parameters)
        mean[..., self._nonperiodic_inds] = mean_nonperiodic
        mean[..., self._periodic_inds] = torch.arctan2(mean_sin_periodic,
                                                       mean_cos_periodic)
        return mean

    def _get_cholesky(self, values):
        """
        Convert a set of ``n_par * (n_par+1) // 2`` values into a lower
        triangular matrix with positive diagonal.
        """
        chol = torch.zeros(
            (values.shape[0], self.n_parameters, self.n_parameters),
            device=values.device)

        # Diagonal, ensuring it's positive:
        inds = np.arange(self.n_parameters)
        chol[:, inds, inds] = torch.exp(values[:, :self.n_parameters])

        # Lower triangle:
        i, j = np.tril_indices(self.n_parameters, -1)
        chol[:, i, j] = values[:, self.n_parameters:]

        return chol

    def _loss_function(self, compressed_data, parameters, sin, cos):
        mean, chol_inv, mean_sin, mean_cos = self._predict_moments(
            compressed_data, ret_mean_sin_cos=True)

        rescaled = self._rescale(parameters, mean, chol_inv)

        chi_squared = (rescaled**2).sum(dim=1)

        eigvals = torch.diagonal(chol_inv, dim1=-2, dim2=-1)
        log_det_chol_inv = torch.log(eigvals).sum(dim=1)

        # This term regularizes the mean_sin and mean_cos of periodic
        # parameters (otherwise, these would enter only through their
        # ratio and have arbitrary norm). It's not derived from a KL
        # divergence, but it should have a similar optimum.
        circular_chisq = ((sin-mean_sin)**2 + (cos-mean_cos)**2).sum(dim=1)

        return torch.mean(chi_squared/2 - log_det_chol_inv + circular_chisq)

    def _get_folded_range_dic(self):
        """
        Return the range_dic of the transform class, setting the value
        for ``'lnq'`` from the config.
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
        Value to be compactified.

    a, b: float
        Bounds of the finite interval.

    Returns
    -------
    float: Compactified value within the interval [a, b].
    """
    return (b - a) / 2 * torch.tanh(value) + (b + a) / 2


def _decompactify(compact_value, a, b):
    """
    Decompactify a value from a finite interval [a, b] to an infinite
    interval using arctanh.

    Parameters
    ----------
    compact_value: float
        Compactified value within the interval [a, b].

    a, b: float
        Bounds of the finite interval.

    Returns
    -------
    float: Decompactified value within the infinite interval.
    """
    return torch.arctanh(2 * (compact_value - (b + a) / 2) / (b - a))


class _MultiLayerPerceptron(nn.Module):
    """Can be saved and loaded without pickle."""
    @classmethod
    def from_dict(cls, dic):
        """
        Load the model's architecture and weights from a dict.

        See Also
        --------
        .to_dict
        """
        model = cls(**dic['architecture'])
        model.load_state_dict(dic['state_dict'])

        return model

    def __init__(self, n_inputs, n_outputs, n_layers=5,
                 layer_size=100, activation_fn=nn.SiLU):
        """
        Parameters
        ----------
        n_inputs: int
            Number of input features.

        n_outputs: int
            Number of output features.

        n_layers: int
            Number of hidden layers.

        layer_size: int
            Number of neurons in each hidden layer.

        activation_fn: nn.Module
            Activation function class from PyTorch (e.g., ``nn.SiLU``)
            without parentheses.
        """
        super().__init__()

        if isinstance(activation_fn, str):
            activation_fn = getattr(nn, activation_fn)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.activation_fn = activation_fn

        layers = []
        layers.append(nn.Linear(n_inputs, layer_size))
        layers.append(activation_fn())

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(activation_fn())

        layers.append(nn.Linear(layer_size, n_outputs))

        self._model = nn.Sequential(*layers)

    def forward(self, x):
        return self._model(x)

    def to_dict(self):
        """
        Return dictionary that can be used to instantiate the class.

        See Also
        --------
        .from_dict
        """
        architecture = {'n_inputs': self.n_inputs,
                        'n_outputs': self.n_outputs,
                        'n_layers': self.n_layers,
                        'layer_size': self.layer_size,
                        'activation_fn': self.activation_fn.__name__}

        return {'architecture': architecture,
                'state_dict': self.state_dict()}


def main(rundir):
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
                rescaled_parameters.detach().cpu().numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Train a model for the mean and covariance and use it to
                       rescale the parameters.''')
    parser.add_argument('rundir', help='Run directory.')

    main(**vars(parser.parse_args()))
