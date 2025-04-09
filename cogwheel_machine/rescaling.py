"""
Rescale physical parameters with a neural network model for the mean and
covariance of the posterior.

To train a model from scratch, use ``main(rescalerdir)``.
To fine-tune an already trained model, you can optionally edit the
relevant rescaler_config parameters and then do:
```
rescaler = ParameterRescaler(rescalerdir)
rescaler.train()
main(rescalerdir)  # Will create the files in the training and test directory
```
"""
import argparse
import cProfile
import logging
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py

import torch
from torch import nn

import cogwheel.utils

from cogwheel_machine import utils, sbi_hacks


logger = logging.getLogger(__name__)

PARAMETER_RESCALER_TRAINING_FILENAME = 'parameter_rescaler_training.pth'
PARAMETER_RESCALER_FILENAME = 'parameter_rescaler.pth'


def plot_loss(rescalerdir):
    """Plot loss function of the rescaling model vs. training epoch."""
    rescalerdir = Path(rescalerdir)
    training_info = torch.load(
        rescalerdir/PARAMETER_RESCALER_TRAINING_FILENAME, weights_only=True)

    plt.figure()
    plt.plot(training_info['train_losses'], label='Training')
    plt.plot(training_info['val_losses'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(linestyle=':')


class ParameterRescaler:
    """
    Rescale physical parameters to make them ~N(0, 1), or unrescale them
    back.

    Bounded parameters are mapped to (-inf, inf) to avoid hard edges.
    Periodic parameters are first centered by predicting their circular
    mean from the data (so as to avoid spurious multimodality when the
    posterior straddles the branch cut) and then mapped to (-inf, inf).

    The top-level function ``main`` provides an interface for this class
    that is suitable for simple use cases.
    """

    def __init__(self, rescalerdir):
        self.rescalerdir = Path(rescalerdir).resolve()
        self.rescaler_config = utils.load_rescaler_config(self.rescalerdir)
        rundir = self.rescalerdir.parents[1]
        self.data_config = utils.load_data_config(rundir)

        self.folded_range_dic = self._get_folded_range_dic()
        self.bounded_params = self._get_bounded_params()

        assert set(self.bounded_params) <= self.folded_range_dic.keys()
        assert set(self.periodic_params) <= self.folded_range_dic.keys()

        device = self.rescaler_config.DEVICE
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                logger.info('cuda unavailable, default to cpu.')
                device = 'cpu'
        self.device = torch.device(device)
        logger.info(f'Using {device=}')

        params = list(self.folded_range_dic)
        self._periodic_inds = [
            params.index(par) for par in self.periodic_params]

        self._bounded_nonperiodic_inds = [
            params.index(par) for par in self.bounded_nonperiodic_params]

        self._nonperiodic_inds = [ind for ind in range(self.n_parameters)
                                  if ind not in self._periodic_inds]

        self._coefs = None  # Set by ._{load|fit}_model
        self._nonperiodic_residuals_scale = None  # Set by ._{load|fit}_model
        self._lnj_scale = None  # Set by ._{load|fit}_model
        self._moments_model = None  # Set by ._{load|fit}_model
        self._training_info = None  # Set by ._{load|fit}_model
        try:
            self._load_model()
        except FileNotFoundError:  # Models have not been trained yet
            logger.info('Did not find existing rescaler, will train one...')

            with cProfile.Profile() as profile:
                self._setup_model()
                self.train()
                profile.dump_stats(self.rescalerdir/'rescaling.profile')

    @property
    def n_parameters(self):
        """Number of parameters."""
        return len(self.folded_range_dic)

    @property
    def periodic_params(self):
        """List of periodic-parameter names."""
        return self.data_config.TRANSFORM_CLASS.periodic_params

    @property
    def bounded_nonperiodic_params(self):
        """List of bounded-non-periodic-parameter names."""
        return [par for par in self.bounded_params
                if par not in self.periodic_params]

    def process_rescalerdir(self):
        """
        Rescale and save parameters in training and test directories.
        """
        rundir = self.rescalerdir.parents[1]

        for datadir in rundir/utils.TRAINING_DIR, rundir/utils.TEST_DIR:
            mask = np.load(datadir/utils.MASK_FILENAME)
            compressed_data = np.load(
                datadir/utils.COMPRESSED_DATA_FILENAME)[mask]

            with h5py.File(datadir/utils.FOLDED_SAMPLED_PARAMETERS_FILENAME,
                           "r") as h5file:
                # The [:] makes it faster
                folded_sampled_parameters = h5file["dataset"][:][mask]

            rescaled_parameters = self.rescale(compressed_data,
                                               folded_sampled_parameters)

            rescaled_datadir = self.rescalerdir/datadir.name
            os.makedirs(rescaled_datadir)

            np.save(rescaled_datadir/utils.RESCALED_PARAMETERS_FILENAME,
                    rescaled_parameters.detach().cpu().numpy())

    def rescale(self, compressed_data, folded_sampled_parameters,
                double_precision=True):
        """
        Apply rescaling to physical parameters to make them ~N(0, 1).

        compressed_data : (n_samples or 1, n_data) float array
            Hint: output of
            ``compression.JSONStandardScaler.transform``, see
            ``compression._save_compressed_data``. # TODO improve docs
            If it contains 1 row, it will broadcast over samples (this
            situation arises in parameter estimation where we have many
            samples for the same data).
            It may also contain a number of rows equal to the number of
            samples, then each sample will use different data (for
            making the training set, where there is one piece of data
            and one true parameters).

        folded_sampled_params : (n_samples, n_params) float array
            Physical parameter values to rescale ("sampled" refers to
            parameters in the domain of the transform, "folded" means
            that folding was applied, to prevent multimodality in the
            distribution).
        """
        compressed_data = torch.as_tensor(compressed_data).to(self.device)
        parameters = torch.as_tensor(folded_sampled_parameters).to(self.device)
        if double_precision:
            parameters = parameters.double()

        model_outputs = self._get_model_outputs(compressed_data)
        mean, chol_inv = self._get_mean_and_chol_inv(model_outputs)

        preconditioned = self._precondition(compressed_data, parameters)
        return self._rescale(preconditioned, mean, chol_inv)

    def _precondition(self, compressed_data, parameters):
        """
        Part of rescaling that does not depend on trainable parameters.
        """
        parameters = parameters.clone().detach()
        self._decompactify_bounded_nonperiodic(parameters)
        self._standardize_nonperiodic(compressed_data, parameters)
        self._periodic_to_angle(parameters)
        return parameters

    def _rescale(self, preconditioned, mean, chol_inv):
        """Part of rescaling that depends on trainable parameters."""
        preconditioned = preconditioned.clone()
        self._remove_mean(mean, preconditioned)
        self._decompactify_periodic(preconditioned)
        rescaled = self._remove_scale(chol_inv, preconditioned)
        return rescaled

    def unrescale(self, compressed_data, rescaled_parameters,
                  double_precision=True):
        """
        Map rescaled parameters from ~N(0, 1) to their physical range.

        Inverse of ``.rescale``.

        Returns
        -------
        parameters : (n_samples, n_params) torch tensor
            Physical parameter values.

        lnj : float
            Log Jacobian determinant of the unrescaling transformation.
        """
        compressed_data = torch.as_tensor(compressed_data).to(self.device)
        parameters = torch.as_tensor(rescaled_parameters).to(self.device)
        if double_precision:
            parameters = parameters.double()

        model_outputs = self._get_model_outputs(compressed_data)
        mean, chol_inv = self._get_mean_and_chol_inv(model_outputs)
        log_det_chol_inv = model_outputs[3].sum(dim=1).detach().numpy()

        return self._unrescale(compressed_data, parameters, mean, chol_inv,
                               log_det_chol_inv)

    def _unrescale(self, compressed_data, parameters, mean, chol_inv,
                   log_det_chol_inv):
        parameters = self._add_scale(chol_inv, parameters)
        lnj = -log_det_chol_inv

        lnj = lnj + self._compactify_periodic(parameters)

        self._add_mean(mean, parameters)

        lnj += self._angle_to_periodic(parameters)

        lnj += self._unstandardize_nonperiodic(compressed_data, parameters)

        lnj += self._compactify_bounded_nonperiodic(parameters)

        return parameters, lnj

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
        lnj = 0.0
        for i, par in zip(self._bounded_nonperiodic_inds,
                          self.bounded_nonperiodic_params):
            parameters[..., i] = _compactify(parameters[..., i],
                                             *self.folded_range_dic[par])
            lnj += _compactify_log_jacobian_determinant(
                parameters[..., i].detach().numpy(),
                *self.folded_range_dic[par])

        return lnj

    def _standardize_nonperiodic(self, compressed_data, parameters):
        """
        Remove a fit and scale inplace from the nonperiodic parameters
        to make them O(1).
        """
        parameters[..., self._nonperiodic_inds] \
            -= self._nonperiodic_fit(compressed_data)

        parameters[..., self._nonperiodic_inds] \
            /= self._nonperiodic_residuals_scale

    def _unstandardize_nonperiodic(self, compressed_data, parameters):
        """
        Add a fit and scale inplace to the nonperiodic parameters, and
        return the log Jacobian determinant.
        """
        parameters[..., self._nonperiodic_inds] \
            *= self._nonperiodic_residuals_scale

        parameters[..., self._nonperiodic_inds] \
            += self._nonperiodic_fit(compressed_data)

        return self._lnj_scale

    def _fit_standardization(self, compressed_data, parameters):
        """
        Fit affine transformation and scale to parameters.

        We standardize the parameters by subtracting a least-squares fit
        and rescaling the residuals by their standard deviation. The fit
        is an affine transformation to the data, of the form

            parameters ≈ (compressed_data|1) @ coefs

        where ``coefs`` is of shape (n_data + 1, n_parameters)
        """
        assert len(parameters) == len(compressed_data)
        assert parameters.ndim == 2
        assert compressed_data.ndim == 2

        parameters = parameters.clone().detach()
        self._decompactify_bounded_nonperiodic(parameters)
        nonperiodic = parameters[:, self._nonperiodic_inds]

        # Add a column of 1 to compressed_data for the affine transformation
        ones = torch.ones((compressed_data.shape[0], 1),
                          device=compressed_data.device)
        data_augmented = torch.hstack([compressed_data, ones])
        logger.info('About to fit coefs')
        self._coefs = torch.linalg.lstsq(data_augmented,
                                         nonperiodic).solution
        logger.info('Done')
        fit = self._nonperiodic_fit(compressed_data)
        self._nonperiodic_residuals_scale = torch.std(nonperiodic - fit,
                                                      dim=0)

    def _nonperiodic_fit(self, compressed_data):
        return compressed_data @ self._coefs[:-1] + self._coefs[-1]

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
        range inplace, return the log Jacobian determinant.

        Inverse of ``._periodic_to_angle``.
        """
        lnj = 0.0
        for i, par in zip(self._periodic_inds, self.periodic_params):
            parameters[..., i], log_abs_slope = self._linear_rescale(
                parameters[..., i],
                (-np.pi, np.pi),
                self.folded_range_dic[par],
                return_lnj=True)
            lnj += log_abs_slope
        return lnj

    @staticmethod
    def _linear_rescale(x, x_rng, y_rng, return_lnj=False):
        slope = (y_rng[1] - y_rng[0]) / (x_rng[1] - x_rng[0])
        rescaled = y_rng[0] + (x - x_rng[0]) * slope
        if return_lnj:
            return rescaled, np.log(np.abs(slope))
        return rescaled


    def _remove_mean(self, mean, parameters):
        """
        Remove mean of the parameters, using circular mean for the
        periodic ones.

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
        Compactify columns for ``.periodic_params`` inplace and return
        the log Jacobian determinant.

        Inverse of ``._decompactify_periodic``.
        """
        lnj = 0.0
        for i in self._periodic_inds:
            parameters[..., i] = _compactify(parameters[..., i], -np.pi, np.pi)
            lnj += _compactify_log_jacobian_determinant(
                parameters.detach()[..., i].numpy(), -np.pi, np.pi)
        return lnj

    def _remove_scale(self, chol_inv, parameters):
        """Divide parameters by their predicted scale."""
        chol_inv = chol_inv.to(parameters.dtype)
        # Same but matmul is faster
        # return torch.einsum('...j,...jk', parameters, chol_inv)
        return torch.matmul(parameters.unsqueeze(-2), chol_inv).squeeze(-2)

    def _add_scale(self, chol_inv, parameters):
        """Multiply parameters by their predicted scale."""
        chol_inv = chol_inv.to(parameters.dtype)
        return torch.linalg.solve_triangular(
            chol_inv, parameters.unsqueeze(-2), upper=False, left=False
            ).squeeze(-2)

    def _load_model(self):
        """
        Set attributes ``_coefs``, ``_nonperiodic_residuals_scale`` and
        ``_moments_model`` by loading from disk.
        """
        model_config = torch.load(self.rescalerdir/PARAMETER_RESCALER_FILENAME,
                                  weights_only=True, map_location=self.device)

        self._coefs = model_config['coefs']

        self._nonperiodic_residuals_scale \
            = model_config['nonperiodic_residuals_scale']
        self._lnj_scale = np.sum(
            np.log(self._nonperiodic_residuals_scale.detach().numpy()))

        self._moments_model = _MultiLayerPerceptron.from_dict(
            model_config['_MultiLayerPerceptron']).to(self.device)

        self._training_info = torch.load(
            self.rescalerdir/PARAMETER_RESCALER_TRAINING_FILENAME,
            weights_only=True, map_location=self.device)

    def _save_current_model(self):
        model_config = {
            '_MultiLayerPerceptron': self._moments_model.to_dict(),
            'coefs': self._coefs,
            'nonperiodic_residuals_scale': self._nonperiodic_residuals_scale}

        torch.save(model_config, self.rescalerdir/PARAMETER_RESCALER_FILENAME)

    def _setup_model(self):
        """
        Set attributes ``_non_periodic_mean`` and ``non_periodic_scale``
        by measuring them from the dataset, and ``_moments_model`` by
        training a neural network.
        """
        compressed_data, parameters, _ = self._load_data()

        n_inputs = compressed_data.shape[1]

        # Mean and Cholesky decomposition of the inverse covariance;
        # periodic parameters require an extra output because we predict
        # the mean sin and cos.
        n_outputs = (self.n_parameters + len(self._periodic_inds)
                     + self.n_parameters * (self.n_parameters + 1) // 2)
        self._moments_model = _MultiLayerPerceptron(
            n_inputs, n_outputs, **self.rescaler_config.RESCALER_NN_KWARGS
            ).to(self.device)

        self._fit_standardization(compressed_data, parameters)

        # TODO could add optimizer state dict
        self._training_info = {'train_losses': [],
                               'val_losses': [],
                               'best_val_loss': np.inf}

    def train(self):
        """
        Train multilayer perceptron model for the mean and covariance.

        This will update the ``._moments_model`` and ``._training_info``
        attributes, and create files with the best model and training
        history in ``.rescalerdir``.
        """
        self._check_no_rescaled_parameter_files()

        kwargs = self.rescaler_config.RESCALER_TRAIN_KWARGS
        optimizer = torch.optim.Adam(self._moments_model.parameters(),
                                     **kwargs['optimizer_kwargs'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **kwargs['scheduler_kwargs'])

        train_loader, val_loader = self._get_dataloaders()

        patience_counter = 0

        try:
            for epoch in range(kwargs['max_num_epochs']):
                # Training
                self._moments_model.train()
                train_loss = 0.0
                for training_batch in train_loader:
                    optimizer.zero_grad()
                    loss = self._loss_function(*training_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                self._training_info['train_losses'].append(train_loss)

                # Validation
                self._moments_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_loader:
                        loss = self._loss_function(*val_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                self._training_info['val_losses'].append(val_loss)

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < self._training_info['best_val_loss']:
                    self._training_info['best_val_loss'] = val_loss
                    patience_counter = 0
                    self._save_current_model()
                else:
                    patience_counter += 1
                    if patience_counter >= kwargs['stop_after_epochs']:
                        break

                print(f'Epoch {epoch} | Validation Loss: {val_loss:.3f}',
                      end='\r')
            print()
        except KeyboardInterrupt:
            print('\nTraining interrupted.')

            if (len(self._training_info['train_losses'])
                    == len(self._training_info['val_losses']) + 1):
                del self._training_info['train_losses'][-1]

            self._moments_model.eval()

        self._save_training_info()
        self._load_model()

    def _check_no_rescaled_parameter_files(self):
        """
        Refuse to retrain the model if there are rescaled parameters
        saved to disk and/or sbi models already trained.
        Otherwise the rescaler would be incorrect and it would be
        impossible to transform back to physical parameters.
        """
        offending_paths = [
            *self.rescalerdir.glob(f'*/{utils.RESCALED_PARAMETERS_FILENAME}'),
            *self.rescalerdir.glob('rescaler_*/')]

        if offending_paths:
            raise RuntimeError(
                'Retraining would make the following files obsolete, '
                'delete them (if you want) and try again.\n'
                + '\n'.join(map(str, offending_paths)))

    def _save_training_info(self):
        torch.save(self._training_info,
                   self.rescalerdir/PARAMETER_RESCALER_TRAINING_FILENAME)
        plot_loss(self.rescalerdir)
        plt.savefig(self.rescalerdir/'rescaling_loss.pdf', bbox_inches='tight')

    def _get_dataloaders(self):
        kwargs = self.rescaler_config.RESCALER_TRAIN_KWARGS

        compressed_data, parameters, weights = self._load_data()
        preconditioned = self._precondition(compressed_data, parameters)
        sin, cos = self._get_sin_cos_periodic_parameters(parameters)
        dataset = torch.utils.data.TensorDataset(
            compressed_data, preconditioned, sin, cos, weights)

        training_batch_size = kwargs['training_batch_size']
        if training_batch_size > (max_size := len(compressed_data) // 10):
            logger.warning('Rescaler batch size too large, reducing it.')
            training_batch_size = max_size

        train_ind_batches, val_ind_batches \
            = sbi_hacks.get_train_val_batch_inds(len(dataset),
                                                 training_batch_size,
                                                 kwargs['validation_fraction'])

        train_batches = [dataset[inds] for inds in train_ind_batches]
        val_batches = [dataset[inds] for inds in val_ind_batches]

        train_loader = sbi_hacks.FixedBatchesDataLoader(train_batches)
        val_loader = sbi_hacks.FixedBatchesDataLoader(val_batches)

        return train_loader, val_loader

    def _load_data(self):
        priordir, rundir = self.rescalerdir.parents[:2]
        datadir = rundir/utils.TRAINING_DIR

        mask = np.load(datadir/utils.MASK_FILENAME)

        logger.info('Loading compressed data...')
        compressed_data = torch.as_tensor(
            np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask],
            device=self.device)

        logger.info('Loading folded sampled parameters...')
        with h5py.File(datadir/utils.FOLDED_SAMPLED_PARAMETERS_FILENAME, "r"
                      ) as h5file:
            # The [:] makes it faster
            parameters = torch.as_tensor(h5file["dataset"][:][mask],
                                         device=self.device)

        weights = torch.as_tensor(
            np.load(priordir/utils.TRAINING_DIR/utils.WEIGHTS_FILENAME),
            device=self.device)

        logger.info('Done')
        return compressed_data, parameters, weights

    def _get_sin_cos_periodic_parameters(self, parameters):
        parameters = parameters.clone().detach()
        self._periodic_to_angle(parameters)
        angle = parameters[:, self._periodic_inds]
        return torch.sin(angle), torch.cos(angle)

    def _get_model_outputs(self, compressed_data):
        """
        Returns
        -------
        torch tensors
            * mean_nonperiodic (n_samples, n_nonperiodic)
            * mean_sin_periodic (n_samples, n_periodic)
            * mean_cos_periodic (n_samples, n_periodic)
            * log_diag_chol_inv (n_samples, n_parameters)
            * offdiagonal_chol_inv (n_samples,
                                    n_parameters*(n_parameters-1)//2)
        """
        output = self._moments_model(compressed_data)

        split_sizes = (len(self._nonperiodic_inds),
                       len(self._periodic_inds),
                       len(self._periodic_inds),
                       self.n_parameters,
                       self.n_parameters * (self.n_parameters - 1) // 2)

        return torch.split(output, split_sizes, dim=1)

    def _get_mean_and_chol_inv(self, model_outputs):
        """
        Predict mean of the parameters, using circular mean for the
        periodic ones.
        """
        (mean_nonperiodic, mean_sin_periodic, mean_cos_periodic,
         log_diag_chol_inv, offdiagonal_chol_inv) = model_outputs

        n_samples, _ = mean_nonperiodic.shape

        # Mean
        mean = torch.empty((n_samples, self.n_parameters), device=self.device)
        mean[..., self._nonperiodic_inds] = mean_nonperiodic
        mean[..., self._periodic_inds] = torch.arctan2(mean_sin_periodic,
                                                       mean_cos_periodic)

        # Cholesky decomposition of the inverse covariance
        chol_inv = torch.zeros(
            (n_samples, self.n_parameters, self.n_parameters),
            device=self.device)
        # - Diagonal, ensuring it's positive:
        inds = np.arange(self.n_parameters)
        chol_inv[:, inds, inds] = torch.exp(log_diag_chol_inv)
        # - Lower triangle:
        i, j = np.tril_indices(self.n_parameters, -1)
        chol_inv[:, i, j] = offdiagonal_chol_inv

        return mean, chol_inv

    def _loss_function(self, compressed_data, preconditioned, sin, cos,
                       weights):
        model_outputs = self._get_model_outputs(compressed_data)
        _, mean_sin_periodic, mean_cos_periodic, log_diag_chol_inv, _ \
            = model_outputs

        mean, chol_inv = self._get_mean_and_chol_inv(model_outputs)
        rescaled = self._rescale(preconditioned.detach(), mean, chol_inv)
        chi_squared = (rescaled**2).sum(dim=1)

        log_det_chol_inv = log_diag_chol_inv.sum(dim=1)

        # This term regularizes the mean_sin and mean_cos of periodic
        # parameters (otherwise, these would enter only through their
        # ratio and have arbitrary norm). It's not derived from a KL
        # divergence, but it should have a similar optimum.
        circular_term = ((sin - mean_sin_periodic) ** 2
                         + (cos - mean_cos_periodic) ** 2
                        ).sum(dim=1)

        return torch.mean(
            weights * (chi_squared/2 - log_det_chol_inv + circular_term))

    def _get_folded_range_dic(self):
        """
        Return the range_dic of the transform class, setting the value
        for ``'lnq'`` from the config.
        """
        # Somewhat fragile, but these methods could be overriden if needed
        folded_range_dic = self.data_config.TRANSFORM_CLASS.range_dic.copy()
        if 'lnq' in folded_range_dic:
            folded_range_dic['lnq'] = (
                np.log(self.data_config.PRIOR_KWARGS['q_min']), 0.0)

        for par in self.data_config.TRANSFORM_CLASS.folded_params:
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
    value : float
        Value to be compactified.

    a, b : float
        Bounds of the finite interval.

    Returns
    -------
    float : Compactified value within the interval [a, b].
    """
    return (b - a) / 2 * torch.tanh(value) + (b + a) / 2


def _decompactify(compact_value, a, b, eps=1e-7):
    """
    Decompactify a value from a finite interval [a, b] to an infinite
    interval using arctanh.

    Parameters
    ----------
    compact_value : float
        Compactified value within the interval [a, b].

    a, b : float
        Bounds of the finite interval.

    eps : float
        Prevents overflow if `compact_value` is close to the edge.

    Returns
    -------
    float : Decompactified value within the infinite interval.
    """
    arg = torch.clamp(2 * (compact_value - (b + a) / 2) / (b - a),
                      -1 + eps, 1 - eps)
    return torch.arctanh(arg)


def _compactify_log_jacobian_determinant(value, a, b):
    """
    Log of the Jacobian determinant of the ``_compactify`` function.

    Parameters
    ----------
    value: float
        The value at which to compute the log Jacobian determinant.

    a, b: float
        The bounds of the finite interval.

    Returns
    -------
    float: The log of the Jacobian determinant.
    """
    return np.log((b - a) / 2) - 2 * _log_cosh(value)


def _log_cosh(x):
    """Numerically stable log(cosh(x))."""
    abs_x = np.abs(x)
    return abs_x + np.log1p(np.exp(-2 * abs_x)) - np.log(2)


class _MultiLayerPerceptron(nn.Module):
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
        n_inputs : int
            Number of input features.

        n_outputs : int
            Number of output features.

        n_layers : int
            Number of hidden layers.

        layer_size : int
            Number of neurons in each hidden layer.

        activation_fn : nn.Module
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
        """Output of the neural network."""
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


def main(rescalerdir):
    """
    Fit mean and scale using a multilayer perceptron, and save rescaled
    parameters.

    This will create files for the model in `rescalerdir` (if not
    already present), and for the rescaled parameters in both the
    training and test directories.
    """
    rescalerdir = Path(rescalerdir)
    logging.basicConfig(filename=rescalerdir/'rescaling.log', encoding='utf-8',
                        level=logging.DEBUG)
    logger.info('Running rescaling')

    parameter_rescaler = ParameterRescaler(rescalerdir)
    parameter_rescaler.process_rescalerdir()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Train a model for the mean and covariance and use it to
                       rescale the parameters.''')
    parser.add_argument('rescalerdir', help='Rescaler directory.')

    main(**vars(parser.parse_args()))
