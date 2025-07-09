"""Functions for training neural posterior estimators."""
import argparse
import logging
import pickle
from pathlib import Path
from cProfile import Profile
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from sbi.inference import NPSE
from sbi.neural_nets import posterior_nn

from cogwheel_machine import compression, embedding, sbi_hacks, utils


logger = logging.getLogger(__name__)


def load_posterior(sbidir, device='cpu'):
    """Load a neural posterior once it has been trained."""
    posterior = torch.load(sbidir/utils.POSTERIOR_FILENAME,
                           map_location=torch.device(device),
                           weights_only=False)
    if hasattr(posterior, '_device'):
        posterior._device = device
    if hasattr(posterior, 'device'):
        posterior.device = device

    return posterior


def load_loss(sbidir):
    """Load the training and validation losses of a trained model."""
    accumulator = _load_accumulator(sbidir)
    training_scalars = accumulator.Scalars('training_loss')
    validation_scalars = accumulator.Scalars('validation_loss')

    training_loss = [scalar.value for scalar in training_scalars]
    validation_loss = [scalar.value for scalar in validation_scalars]
    epochs = [scalar.step for scalar in training_scalars]

    return epochs, training_loss, validation_loss


def load_runtime(sbidir):
    """Array of length n_epochs with cumulative training time (h)."""
    accumulator = _load_accumulator(sbidir)
    durations = [x.value for x in accumulator.Scalars('epoch_durations_sec')]
    return np.cumsum(durations) / 3600


def _load_accumulator(sbidir):
    """Return tensorboard EventAccumulator saved in `sbidir`."""
    accumulator = event_accumulator.EventAccumulator(
        str(sbidir),
        size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE)
    accumulator.Reload()
    return accumulator


def plot_loss(sbidir, save=True):
    """Plot the training and validation losses of a trained model."""
    epochs, training_loss, validation_loss = load_loss(sbidir)

    plt.figure()
    plt.plot(epochs, training_loss, label='Training')
    plt.plot(epochs, validation_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(ls=':')
    plt.title(sbidir.name)

    if save:
        plt.savefig(sbidir/'loss.pdf', bbox_inches='tight')


def _instantiate_inference(sbidir):
    rescalerdir, priordir, rundir = sbidir.resolve().parents[:3]
    datadir = rundir/utils.TRAINING_DIR
    rescaled_datadir = rescalerdir/utils.TRAINING_DIR
    config = utils.load_sbi_config(sbidir)

    device = config.DEVICE
    if device is None:
        device = utils.get_best_device()
        logger.info(f'Using {device=}')

    mask = np.load(datadir/utils.MASK_FILENAME)

    rescaled_parameters = np.load(
        rescaled_datadir/utils.RESCALED_PARAMETERS_FILENAME
        )[:config.MAX_TRAINING_EXAMPLES]

    simulation_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME
                             )[mask][:config.MAX_TRAINING_EXAMPLES]

    simulation_weights = np.load(
        priordir/utils.TRAINING_DIR/utils.WEIGHTS_FILENAME
        )[:config.MAX_TRAINING_EXAMPLES]

    theta = torch.tensor(rescaled_parameters, dtype=torch.float32).to(device)
    x = torch.tensor(simulation_data, dtype=torch.float32).to(device)
    weights = torch.tensor(simulation_weights, dtype=torch.float32).to(device)

    if config.EMBEDDING_LAYER_SIZES:
        embedding_net = embedding.BlockMatrixEmbeddingNetwork(
            input_size=x.shape[1],
            layer_sizes=config.EMBEDDING_LAYER_SIZES,
            unchanged_size=_get_n_processed_coef(rundir))
        config.POSTERIOR_NN_KWARGS['embedding_net'] = embedding_net

    if config.POSTERIOR_NN_KWARGS:
        neural_posterior = posterior_nn(**config.POSTERIOR_NN_KWARGS)

        inference = sbi_hacks.NPEFixedBatches(
            density_estimator=neural_posterior,
            device=device,
            summary_writer=SummaryWriter(sbidir)
            ).append_simulations(theta, x, weights=weights)
        
    elif config.SCORE_NN_KWARGS:

        inference = NPSE(**config.SCORE_NN_KWARGS,
            device=device,
            summary_writer=SummaryWriter(sbidir)
            ).append_simulations(theta, x, weights=weights)

    return inference


def _get_n_processed_coef(rundir):
    _, n_total = np.load(
        rundir/utils.TEST_DIR/utils.COMPRESSED_DATA_FILENAME).shape
    n_svd = compression.SVDCompressor.from_npz(rundir).n_components
    return n_total - n_svd


def main(sbidir):
    """
    Train neural posterior estimator.

    Parameters
    ----------
    sbidir : os.PathLike
        Path to directory inside a ``rundir``, containing a file
        "sbi_config.py". If `sbidir` also contains a previously
        trained model, it will resume training.

    See Also
    --------
    utils.setup_sbidir
    """
    sbidir = Path(sbidir)

    logging.basicConfig(filename=sbidir/'training.log', encoding='utf-8',
                        level=logging.DEBUG)
    logger.info('Running training')

    config = utils.load_sbi_config(sbidir)

    inference_filename = sbidir/utils.INFERENCE_FILENAME
    resume_training = inference_filename.exists()
    if resume_training:
        with open(inference_filename, 'rb') as file:
            inference = pickle.load(file)
    else:
        inference = _instantiate_inference(sbidir)

    with Profile() as profiler:
        density_estimator = inference.train(
            **config.TRAIN_KWARGS,
            resume_training=resume_training,
            force_first_round_loss=resume_training,
            lr_scheduler_kwargs=config.LR_SCHEDULER_KWARGS)

    profiler.dump_stats(sbidir/'training.profile')

    with open(inference_filename, 'wb') as file:
        pickle.dump(inference, file)

    posterior = inference.build_posterior(density_estimator)
    torch.save(posterior, sbidir/utils.POSTERIOR_FILENAME)
    plot_loss(sbidir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train neural posterior estimator from existing data')
    parser.add_argument('sbidir',
                        help='''Path of the run directory, must contain a
                                (populated) `training_data/` subdirectory.''')

    main(**vars(parser.parse_args()))
