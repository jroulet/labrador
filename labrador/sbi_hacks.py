"""Modifications to the behavior of ``sbi``."""
import functools
import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
import numpy as np
from torch import Tensor
import torch.utils.data
from sbi.utils.sbiutils import get_simulations_since_round, ImproperEmpirical
from sbi.inference.trainers.npe.npe_base import PosteriorEstimator

from sbi.inference.trainers.npe.npe_base import (
    Adam,
    ConditionalDensityEstimator,
    clip_grad_norm_,
    deepcopy,
    ones,
    reshape_to_sample_batch_event,
    reshape_to_batch_event,
    test_posterior_net_for_multi_d_x,
    time,
)


class NPEFixedBatches(PosteriorEstimator):
    """
    Like sbi.inference.NPE except the batches are fixed.

    The batches are made of consecutive simulations (no shuffling), the
    first batches are training and the last are validation.

    By making the simulations consecutive we try to preserve the low-
    discrepancy property of QMC sequences.
    By making the batches once and for all we speed up iterations over
    the data.
    """
    @functools.wraps(PosteriorEstimator.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weights_roundwise = []

    def get_simulations(
        self,
        starting_round: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""Returns all $\theta$, $x$, prior_masks and weights from
        rounds >= `starting_round`.

        If requested, do not return invalid data.

        Args:
            starting_round: The earliest round to return samples from
            (we start counting from zero).
            warn_on_invalid: Whether to give out a warning if invalid
            simulations were found.

        Returns: Parameters, simulation outputs, prior masks and weights.
        """
        theta, x, prior_masks = super().get_simulations(starting_round)
        weights = get_simulations_since_round(
            self._weights_roundwise, self._data_round_index, starting_round
        )

        return theta, x, prior_masks, weights

    def get_dataloaders(self,
                        starting_round: int = 0,
                        training_batch_size: int = 200,
                        validation_fraction: float = 0.1,
                        resume_training=None,
                        dataloader_kwargs=None):
        """Return dataloaders for training and validation."""
        if dataloader_kwargs:
            print(f'Ignoring `{dataloader_kwargs=}`')

        dataset = torch.utils.data.TensorDataset(
            *self.get_simulations(starting_round))

        if not resume_training:
            # These allow to preserve the exact partition into batches
            # as well as training/validation over multiple trainings.
            self._train_ind_batches, self._val_ind_batches \
                = get_train_val_batch_inds(len(dataset), training_batch_size,
                                           validation_fraction)

            # Other methods assume this attribute exists
            self.train_indices = np.concatenate(self._train_ind_batches)

        train_batches = [dataset[inds] for inds in self._train_ind_batches]
        val_batches = [dataset[inds] for inds in self._val_ind_batches]

        train_loader = FixedBatchesDataLoader(train_batches)
        val_loader = FixedBatchesDataLoader(val_batches)

        return train_loader, val_loader

    def append_simulations(self, theta, x, proposal=None,
                           exclude_invalid_x=None, data_device=None, *,
                           weights=None):
        """
        Append simulations, including weights

        Like sbi.inference.trainers.npe.npe_base.PosteriorEstimator.append_simulations
        but it also appends weights.
        """
        if weights is None:
            weights = torch.ones(len(theta))
        self._weights_roundwise.append(weights)

        super().append_simulations(theta, x, proposal, exclude_invalid_x,
                                   data_device)

        # Patch: an ImproperEmpiricalPrior with more than 2^24
        # categories crashes when calling `.sample`.
        max_samples = 2**24
        if (isinstance(self._prior, ImproperEmpirical)
                and self._prior.sample_size > max_samples):
            self._prior = ImproperEmpirical(
                self._prior._samples[:max_samples].clone(),
                self._prior._log_weights[:max_samples].clone()
            )  # Clone to save the memory of the truncated part

        return self

    # Override ``train`` method to allow `lr_scheduler_kwargs`.
    # The code below is copied from sbi.inference.trainers.npe.npe_base
    # almost verbatim, the new lines w.r.t the sbi implementation have
    # an asterisk "# *".
    # pylint: disable=line-too-long
    def train(
        self,
        training_batch_size: int = 200,
        learning_rate: float = 5e-4,
        validation_fraction: float = 0.1,
        stop_after_epochs: int = 20,
        max_num_epochs: int = 2**31 - 1,
        clip_max_norm: Optional[float] = 5.0,
        calibration_kernel: Optional[Callable] = None,
        resume_training: bool = False,
        force_first_round_loss: bool = False,
        discard_prior_samples: bool = False,
        retrain_from_scratch: bool = False,
        show_train_summary: bool = False,
        dataloader_kwargs: Optional[dict] = None,
        lr_scheduler_kwargs=None,  # *
    ) -> ConditionalDensityEstimator:
        r"""Return density estimator that approximates the distribution $p(\theta|x)$.

        Args:
            training_batch_size: Training batch size.
            learning_rate: Learning rate for Adam optimizer.
            validation_fraction: The fraction of data to use for validation.
            stop_after_epochs: The number of epochs to wait for improvement on the
                validation set before terminating training.
            max_num_epochs: Maximum number of epochs to run. If reached, we stop
                training even when the validation loss is still decreasing. Otherwise,
                we train until validation loss increases (see also `stop_after_epochs`).
            clip_max_norm: Value at which to clip the total gradient norm in order to
                prevent exploding gradients. Use None for no clipping.
            calibration_kernel: A function to calibrate the loss with respect
                to the simulations `x` (optional). See Lueckmann, Gonçalves et al.,
                NeurIPS 2017. If `None`, no calibration is used.
            resume_training: Can be used in case training time is limited, e.g. on a
                cluster. If `True`, the split between train and validation set, the
                optimizer, the number of epochs, and the best validation log-prob will
                be restored from the last time `.train()` was called.
            force_first_round_loss: If `True`, train with maximum likelihood,
                i.e., potentially ignoring the correction for using a proposal
                distribution different from the prior.
            discard_prior_samples: Whether to discard samples simulated in round 1, i.e.
                from the prior. Training may be sped up by ignoring such less targeted
                samples.
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            show_train_summary: Whether to print the number of epochs and validation
                loss after the training.
            dataloader_kwargs: Additional or updated kwargs to be passed to the training
                and validation dataloaders (like, e.g., a collate_fn)
            lr_scheduler_kwargs: passed to torch.optim.lr_scheduler.ReduceLROnPlateau  # *

        Returns:
            Density estimator that approximates the distribution $p(\theta|x)$.
        """
        # Load data from most recent round.
        self._round = max(self._data_round_index)

        if self._round == 0 and self._neural_net is not None:
            assert force_first_round_loss or resume_training, (
                "You have already trained this neural network. After you had trained "
                "the network, you again appended simulations with `append_simulations"
                "(theta, x)`, but you did not provide a proposal. If the new "
                "simulations are sampled from the prior, you can set "
                "`.train(..., force_first_round_loss=True`). However, if the new "
                "simulations were not sampled from the prior, you should pass the "
                "proposal, i.e. `append_simulations(theta, x, proposal)`. If "
                "your samples are not sampled from the prior and you do not pass a "
                "proposal and you set `force_first_round_loss=True`, the result of "
                "SNPE will not be the true posterior. Instead, it will be the proposal "
                "posterior, which (usually) is more narrow than the true posterior."
            )

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:

            def default_calibration_kernel(x):
                return ones([len(x)], device=self._device)

            calibration_kernel = default_calibration_kernel

        # Starting index for the training set (1 = discard round-0 samples).
        start_idx = int(discard_prior_samples and self._round > 0)

        # For non-atomic loss, we can not reuse samples from previous rounds as of now.
        # SNPE-A can, by construction of the algorithm, only use samples from the last
        # round. SNPE-A is the only algorithm that has an attribute `_ran_final_round`,
        # so this is how we check for whether or not we are using SNPE-A.
        if self.use_non_atomic_loss or hasattr(self, "_ran_final_round"):
            start_idx = self._round

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            training_batch_size,
            validation_fraction,
            resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )
        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, *_ = self.get_simulations(starting_round=start_idx)
            # Use only training data for building the neural net (z-scoring transforms)

            self._neural_net = self._build_neural_net(
                theta[self.train_indices].to("cpu"),
                x[self.train_indices].to("cpu"),
            )

            theta = reshape_to_sample_batch_event(
                theta.to("cpu"), self._neural_net.input_shape
            )
            x = reshape_to_batch_event(x.to("cpu"), self._neural_net.condition_shape)
            test_posterior_net_for_multi_d_x(self._neural_net, theta, x)

            del theta, x

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if not resume_training:
            self.optimizer = Adam(list(self._neural_net.parameters()), lr=learning_rate)
            self.epoch, self._val_loss = 0, float("Inf")

        if lr_scheduler_kwargs is not None:  # *
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # *
                self.optimizer, **lr_scheduler_kwargs)  # *

        while self.epoch <= max_num_epochs and not self._converged(
            self.epoch, stop_after_epochs
        ):
            # Train for a single epoch.
            self._neural_net.train()
            train_loss_sum = 0
            epoch_start_time = time.time()
            for batch in train_loader:
                self.optimizer.zero_grad()
                # Get batches on current device.
                theta_batch, x_batch, masks_batch, weights_batch = (
                    batch[0].to(self._device),
                    batch[1].to(self._device),
                    batch[2].to(self._device),
                    batch[3].to(self._device),
                )

                train_losses = self._loss(
                    theta_batch,
                    x_batch,
                    masks_batch,
                    proposal,
                    calibration_kernel,
                    force_first_round_loss=force_first_round_loss,
                    weights=weights_batch,
                )
                train_loss = torch.mean(train_losses)
                train_loss_sum += train_losses.sum().item()

                train_loss.backward()
                if clip_max_norm is not None:
                    clip_grad_norm_(
                        self._neural_net.parameters(), max_norm=clip_max_norm
                    )
                self.optimizer.step()

            self.epoch += 1

            train_loss_average = train_loss_sum / (
                len(train_loader) * train_loader.batch_size  # type: ignore
            )
            self._summary["training_loss"].append(train_loss_average)

            # Calculate validation performance.
            self._neural_net.eval()
            val_loss_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    theta_batch, x_batch, masks_batch, weights_batch = (
                        batch[0].to(self._device),
                        batch[1].to(self._device),
                        batch[2].to(self._device),
                        batch[3].to(self._device),
                    )
                    # Take negative loss here to get validation log_prob.
                    val_losses = self._loss(
                        theta_batch,
                        x_batch,
                        masks_batch,
                        proposal,
                        calibration_kernel,
                        force_first_round_loss=force_first_round_loss,
                        weights=weights_batch,
                    )
                    val_loss_sum += val_losses.sum().item()

            # Take mean over all validation samples.
            self._val_loss = val_loss_sum / (
                len(val_loader) * val_loader.batch_size  # type: ignore
            )
            # Log validation loss for every epoch.
            self._summary["validation_loss"].append(self._val_loss)
            self._summary["epoch_durations_sec"].append(time.time() - epoch_start_time)

            self._maybe_show_progress(self._show_progress_bars, self.epoch)

            if hasattr(self, 'lr_scheduler'):  # *
                self.lr_scheduler.step(self._val_loss)  # *

        self._report_convergence_at_end(self.epoch, stop_after_epochs, max_num_epochs)

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._best_val_loss)

        # Update tensorboard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return deepcopy(self._neural_net)

    def _loss(self, theta, x, masks, proposal, calibration_kernel,
              force_first_round_loss, weights):
        return weights * super()._loss(theta, x, masks, proposal,
                                       calibration_kernel,
                                       force_first_round_loss)

    def _log_prob_proposal_posterior(
        self,
        theta: Tensor,
        x: Tensor,
        masks: Tensor,
        proposal: Optional[Any],
    ) -> Tensor:
        raise NotImplementedError(
            "Sequential posterior estimation not implemented.")


    def _save_progress(self):
        """Save summary and model."""
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._best_val_loss)
        self._summarize(round_=self._round)

        inference_filename = Path(
            self._summary_writer.log_dir)/'inference_test.pickle'
        with open(inference_filename, 'wb') as file:
            pickle.dump(self, file)


def get_train_val_batch_inds(num_simulations, training_batch_size,
                             validation_fraction):
    """
    Returns
    -------
    train_ind_batches, val_ind_batches : list of int arrays
        Indices of the training and validation simulations, arranged in
        batches.
    """
    num_batches = num_simulations // training_batch_size
    batch_indices = np.split(np.arange(num_batches * training_batch_size),
                             num_batches)

    num_training_batches = int(
        len(batch_indices) * (1-validation_fraction))
    train_ind_batches = batch_indices[:num_training_batches]
    val_ind_batches = batch_indices[num_training_batches:]
    return train_ind_batches, val_ind_batches


class FixedBatchesDataLoader:
    """A list of batches, always the same."""
    def __init__(self, batches, shuffle_batches=True):
        """
        Parameters
        ----------
        batches : list of lists of torch.Tensor
            Each batch contains multiple tensors, e.g. data and
            parameters.

        shuffle_batches : bool
            Whether to iterate over the batches in random order every
            time.
        """
        if len(set(map(len, batches))) != 1:
            raise ValueError('Batches are not the same size.')

        self.batches = batches
        self.shuffle_batches = shuffle_batches

        self.batch_size = len(batches[0][0])
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        order = np.arange(len(self.batches))
        if self.shuffle_batches:
            self._rng.shuffle(order)

        for i in order:
            yield self.batches[i]
