"""Functions for training neural posterior estimators."""
import argparse
import pickle
from pathlib import Path
from cProfile import Profile
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

import sbi.utils

from cogwheel_machine import embedding, sbi_hacks, utils


def load_posterior(modeldir, device='cpu'):
    """Load a neural posterior once it has been trained."""
    posterior = torch.load(modeldir/utils.POSTERIOR_FILENAME,
                           map_location=torch.device(device),
                           weights_only=False)
    posterior._device = device
    return posterior


def load_loss(modeldir):
    """Load the training and validation losses of a trained model."""
    accumulator = event_accumulator.EventAccumulator(
        modeldir.as_posix(),
        size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE
    )
    accumulator.Reload()

    training_scalars = accumulator.Scalars('training_loss')
    validation_scalars = accumulator.Scalars('validation_loss')

    training_loss = [scalar.value for scalar in training_scalars]
    validation_loss = [scalar.value for scalar in validation_scalars]
    epochs = [scalar.step for scalar in training_scalars]

    return epochs, training_loss, validation_loss


def load_runtime(modeldir):
    """Array of length n_epochs with cumulative training time (h)."""
    accumulator = event_accumulator.EventAccumulator(
        modeldir.as_posix(),
        size_guidance=event_accumulator.STORE_EVERYTHING_SIZE_GUIDANCE)
    accumulator.Reload()
    durations = [x.value for x in accumulator.Scalars('epoch_durations_sec')]
    return np.cumsum(durations) / 3600


def plot_loss(modeldir, save=True):
    """Plot the training and validation losses of a trained model."""
    epochs, training_loss, validation_loss = load_loss(modeldir)

    plt.figure()
    plt.plot(epochs, training_loss, label='Training')
    plt.plot(epochs, validation_loss, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(ls=':')
    plt.title(modeldir.name)

    if save:
        plt.savefig(modeldir/'loss.pdf', bbox_inches='tight')


def _instantiate_inference(modeldir):
    datadir = modeldir.resolve().parent/utils.TRAINING_DIR
    config = utils.load_model_config(modeldir)

    mask = np.load(datadir/utils.MASK_FILENAME)

    simulation_parameters = np.load(
        datadir/utils.RESCALED_PARAMETERS_FILENAME
        )[:config.MAX_TRAINING_EXAMPLES]

    simulation_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME
                             )[mask][:config.MAX_TRAINING_EXAMPLES]

    theta = torch.tensor(simulation_parameters, dtype=torch.float32
                        ).to(config.DEVICE)
    x = torch.tensor(simulation_data, dtype=torch.float32).to(config.DEVICE)

    with np.load(datadir/utils.PREPROCESSED_DATA_FILENAME) as file:
        n_processed_coef = file['processed_coef'].shape[1]

    if config.EMBEDDING_LAYER_SIZES:
        embedding_net = embedding.BlockMatrixEmbeddingNetwork(
            input_size=x.shape[1],
            layer_sizes=config.EMBEDDING_LAYER_SIZES,
            unchanged_size=n_processed_coef)
        config.POSTERIOR_NN_KWARGS['embedding_net'] = embedding_net

    neural_posterior = sbi.utils.posterior_nn(**config.POSTERIOR_NN_KWARGS)

    inference = sbi_hacks.NPEFixedBatches(
        density_estimator=neural_posterior,
        device=config.DEVICE,
        summary_writer=SummaryWriter(modeldir)
        ).append_simulations(theta, x)
    return inference


def main(modeldir):
    """
    Train neural posterior estimator.

    Parameters
    ----------
    modeldir: os.PathLike
        Path to directory inside a ``rundir``, containing a file
        "model_config.py". If `modeldir` also contains a previously
        trained model, it will resume training.

    See Also
    --------
    utils.setup_modeldir
    """
    modeldir = Path(modeldir)
    config = utils.load_model_config(modeldir)

    inference_filename = modeldir/utils.INFERENCE_FILENAME
    resume_training = inference_filename.exists()
    if resume_training:
        with open(inference_filename, 'rb') as file:
            inference = pickle.load(file)
    else:
        inference = _instantiate_inference(modeldir)

    with Profile() as profiler:
        density_estimator = inference.train(
            **config.TRAIN_KWARGS,
            resume_training=resume_training,
            force_first_round_loss=resume_training)

    profiler.dump_stats(modeldir/'profiling')

    with open(inference_filename, 'wb') as file:
        pickle.dump(inference, file)

    posterior = inference.build_posterior(density_estimator)
    torch.save(posterior, modeldir/utils.POSTERIOR_FILENAME)
    plot_loss(modeldir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train neural posterior estimator from existing data')
    parser.add_argument('modeldir',
                        help='''Path of the run directory, must contain a
                                (populated) `training_data/` subdirectory.''')

    main(**vars(parser.parse_args()))
