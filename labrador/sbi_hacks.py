"""Modifications to the behavior of ``sbi``."""
import functools
import time
from dataclasses import dataclass

from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import numpy as np

from torch import Tensor
import torch.utils.data
from torch.optim import Adam

from sbi.utils.sbiutils import get_simulations_since_round
from sbi.inference.trainers.npe.npe_base import (
    ConditionalDensityEstimator,
    StartIndexContext,
    LossArgsNPE,
    PosteriorEstimatorTrainer,
    LossArgs,
    ones,
)
from sbi.inference.posteriors.direct_posterior import DirectPosterior
import sbi.inference.trainers._contracts
from sbi.inference.trainers.npe import NPE
from sbi.inference.trainers.base import NeuralInference
from sbi.neural_nets.estimators.base import ConditionalEstimatorType
from sbi.neural_nets.estimators.shape_handling import (
    reshape_to_batch_event,
    reshape_to_sample_batch_event,
)
from sbi.utils.user_input_checks import test_posterior_net_for_multi_d_x
# pylint: disable=line-too-long


# ----------------------------------------------------------------------
# Implement fixed batches, useful for QMC

class PosteriorEstimatorTrainerFixedBatchesMixin:
    """
    Like sbi.inference.trainers.npe.npe_base.PosteriorEstimatorTrainer
    except the batches are fixed.

    The batches are made of consecutive simulations (no shuffling), the
    first batches are training and the last are validation.

    By making the simulations consecutive we try to preserve the low-
    discrepancy property of QMC sequences.
    By making the batches once and for all we speed up iterations over
    the data.
    """
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

        # These allow to preserve the exact partition into batches as
        # well as training/validation over multiple trainings.
        if not resume_training:
            self._train_ind_batches, self._val_ind_batches \
                = get_train_val_batch_inds(len(dataset), training_batch_size,
                                           validation_fraction)

            # Other methods assume these attributes exist
            self.train_indices = np.concatenate(self._train_ind_batches)
            self.val_indices = np.concatenate(self._val_ind_batches)

        train_batches = [dataset[inds] for inds in self._train_ind_batches]
        val_batches = [dataset[inds] for inds in self._val_ind_batches]

        train_loader = FixedBatchesDataLoader(train_batches)
        val_loader = FixedBatchesDataLoader(val_batches)

        return train_loader, val_loader


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
    def __init__(self, batches):
        """
        Parameters
        ----------
        batches : list of lists of torch.Tensor
            Each batch contains multiple tensors, e.g. data and
            parameters.
        """
        if len(set(map(len, batches))) != 1:
            raise ValueError('Batches are not the same size.')

        self.batches = batches
        self.batch_size = len(batches[0][0])

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        yield from self.batches


# ----------------------------------------------------------------------
# Implement counterweight method (user must compute the weights)

class NPECounterWeight(NPE):
    @functools.wraps(NPE.__init__)
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

    def append_simulations(self, theta, x, proposal=None,
                           exclude_invalid_x=None, data_device=None, *,
                           weights=None):
        """
        Append simulations, including weights

        Like sbi.inference.trainers.npe.npe_base.PosteriorEstimatorTrainer.append_simulations
        but it also appends weights.
        """
        if weights is None:
            weights = torch.ones(len(theta))
        self._weights_roundwise.append(weights)

        return super().append_simulations(
            theta, x, proposal, exclude_invalid_x, data_device)

    def _get_losses(self, batch: Sequence[Tensor], loss_args: LossArgs) -> Tensor:
        """
        Compute losses for a batch of data.

        Args:
            batch: A batch of data.
            loss_args: Additional arguments passed to self._loss fn.

        Returns:
            A tensor containing the computed losses for each sample in the batch.
        """
        weights_batch = batch[3].to(self._device)
        return weights_batch * super()._get_losses(batch, loss_args)

    def _initialize_neural_network(
        self,
        retrain_from_scratch: bool,
        start_idx: int,
    ) -> None:
        """
        Initialize the neural network if it is None or retraining from scratch.

        Args:
            retrain_from_scratch: Whether to retrain the conditional density
                estimator for the posterior from scratch each round.
            start_idx: The index of the first round to retrieve simulation data from.
        """

        # First round or if retraining from scratch:
        # Call the `self._build_neural_net` with the rounds' thetas and xs as
        # arguments, which will build the neural network.
        # This is passed into NeuralPosterior, to create a neural posterior which
        # can `sample()` and `log_prob()`. The network is accessible via `.net`.
        if self._neural_net is None or retrain_from_scratch:
            # Get theta,x to initialize NN
            theta, x, _, _ = self.get_simulations(starting_round=start_idx)
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

    def _log_prob_proposal_posterior(
            self,
            theta: Tensor,
            x: Tensor,
            masks: Tensor,
            proposal: DirectPosterior,
    ) -> Tensor:
        raise NotImplementedError(
            "Sequential posterior estimation not implemented.")


# ----------------------------------------------------------------------
# Implement learning rate scheduler


@dataclass
class TrainConfig(sbi.inference.trainers._contracts.TrainConfig):
    lr_scheduler_kwargs: Optional[dict] = None


class PosteriorEstimatorTrainerLRSchedulerMixin:
    """Implement LRScheduler on PosteriorEstimatorTrainer."""

    # Note: `train` copied verbatim from
    # sbi.sbi.inference.trainers.npe.npe_base.PosteriorEstimatorTrainer
    # except for adding lr_scheduler.
    # Added/modified lines have a "  # *".
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
        lr_scheduler_kwargs: Optional[dict] = None,  # *
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

        if len(self._data_round_index) == 0:
            raise RuntimeError(
                "No simulations found. You must call .append_simulations() "
                "before calling .train()."
            )

        train_config = TrainConfig(
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            learning_rate=learning_rate,
            resume_training=resume_training,
            show_train_summary=show_train_summary,
            training_batch_size=training_batch_size,
            retrain_from_scratch=retrain_from_scratch,
            validation_fraction=validation_fraction,
            clip_max_norm=clip_max_norm,
            lr_scheduler_kwargs=lr_scheduler_kwargs,  # *
        )

        # Calibration kernels proposed in Lueckmann, Gonçalves et al., 2017.
        if calibration_kernel is None:

            def default_calibration_kernel(x):
                return ones([len(x)], device=self._device)

            calibration_kernel = default_calibration_kernel

        start_idx = self._get_start_index(
            context=StartIndexContext(
                discard_prior_samples=discard_prior_samples,
                force_first_round_loss=force_first_round_loss,
                resume_training=train_config.resume_training,
            )
        )

        # Set the proposal to the last proposal that was passed by the user. For
        # atomic SNPE, it does not matter what the proposal is. For non-atomic
        # SNPE, we only use the latest data that was passed, i.e. the one from the
        # last proposal.
        proposal = self._proposal_roundwise[-1]

        train_loader, val_loader = self.get_dataloaders(
            start_idx,
            train_config.training_batch_size,
            train_config.validation_fraction,
            train_config.resume_training,
            dataloader_kwargs=dataloader_kwargs,
        )

        self._initialize_neural_network(
            retrain_from_scratch=train_config.retrain_from_scratch,
            start_idx=start_idx,
        )

        loss_args = LossArgsNPE(
            proposal=proposal,
            calibration_kernel=calibration_kernel,
            force_first_round_loss=force_first_round_loss,
        )

        return self._run_training_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            train_config=train_config,
            loss_args=loss_args,
        )


    # Note: `_run_training_loop` copied verbatim from
    # sbi.sbi.inference.trainers.base.NeuralInference except
    # for adding lr_scheduler.
    # Added/modified lines have a "  # *".
    def _run_training_loop(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        train_config: TrainConfig,
        loss_args: LossArgs | None = None,
        summarization_kwargs: Optional[Dict[str, Any]] = None,
    ) -> ConditionalEstimatorType:
        """
        Run the main training loop for the neural network, including epoch-wise
        training, validation, and convergence checking.

        Args:
            train_loader: Dataloader for training.
            val_loader: Dataloader for validation.
            train_config: TrainConfig dataclass configuration for the core training
                path.
            loss_args: Additional arguments passed to self._loss fn.
            summarization_kwargs: Additional kwargs passed to self._summarize_epoch fn.
        """

        if summarization_kwargs is None:
            summarization_kwargs = {}

        assert self._neural_net is not None

        # Move entire net to device for training.
        self._neural_net.to(self._device)

        if not train_config.resume_training:
            self.optimizer = Adam(
                list(self._neural_net.parameters()),
                lr=train_config.learning_rate,
            )
            self.epoch, self.val_loss = 0, float("Inf")

        if getattr(train_config, 'lr_scheduler_kwargs', None) is not None:  # *
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # *
                self.optimizer, **train_config.lr_scheduler_kwargs)  # *

        while self.epoch <= train_config.max_num_epochs and not self._converged(
            self.epoch, train_config.stop_after_epochs
        ):
            # Train for a single epoch.
            self._neural_net.train()
            epoch_start_time = time.time()
            train_loss = self._train_epoch(
                train_loader, train_config.clip_max_norm, loss_args
            )

            # Calculate validation performance.
            self._neural_net.eval()

            self._val_loss = self._validate_epoch(val_loader, loss_args)

            self._summarize_epoch(
                train_loss, self._val_loss, epoch_start_time, summarization_kwargs
            )

            self.epoch += 1
            self._maybe_show_progress(self._show_progress_bars, self.epoch)

            if hasattr(self, 'lr_scheduler'):  # *
                self.lr_scheduler.step(self._val_loss)  # *

        self._report_convergence_at_end(
            self.epoch, train_config.stop_after_epochs, train_config.max_num_epochs
        )

        # Update summary.
        self._summary["epochs_trained"].append(self.epoch)
        self._summary["best_validation_loss"].append(self._best_val_loss)

        # Update TensorBoard and summary dict.
        self._summarize(round_=self._round)

        # Update description for progress bar.
        if train_config.show_train_summary:
            print(self._describe_round(self._round, self._summary))

        # Avoid keeping the gradients in the resulting network, which can
        # cause memory leakage when benchmarking.
        self._neural_net.zero_grad(set_to_none=True)

        return self._neural_net


# ----------------------------------------------------------------------
# Combine everything

class LabradorNPE(PosteriorEstimatorTrainerLRSchedulerMixin,
                  PosteriorEstimatorTrainerFixedBatchesMixin,
                  NPECounterWeight,
                  ):
    pass
