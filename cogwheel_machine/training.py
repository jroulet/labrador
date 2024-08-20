"""Functions for training neural posterior estimators."""
import argparse
from pathlib import Path
from cProfile import Profile
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from sbi.inference import SNPE
import sbi.utils

from cogwheel_machine import utils


def load_logprob(modeldir):
    """
    Load the training and validation log probabilities of a trained
    model.

    The log probabilities are minus the loss.
    """
    accumulator = event_accumulator.EventAccumulator(modeldir.as_posix())
    accumulator.Reload()

    training_logprob = [
        logprob.value for logprob in accumulator.Scalars('training_log_probs')]
    validation_logprob = [
        logprob.value
        for logprob in accumulator.Scalars('validation_log_probs')]
    return training_logprob, validation_logprob


def load_runtime(modeldir):
    """Array of length n_epochs with cumulative training time (h)."""
    accumulator = event_accumulator.EventAccumulator(modeldir.as_posix())
    accumulator.Reload()
    durations = [x.value for x in accumulator.Scalars('epoch_durations_sec')]
    return np.cumsum(durations) / 3600


def plot_logprob(modeldir, save=True):
    """
    Plot the training and validation log probabilities of a trained
    model.
    """
    training_logprob, validation_logprob = load_logprob(modeldir)

    plt.figure()
    plt.plot(training_logprob, label='Training')
    plt.plot(validation_logprob, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Log Prob')
    plt.legend()
    plt.grid(ls=':')
    plt.title(modeldir.name)

    if save:
        plt.savefig(modeldir/'logprob.pdf', bbox_inches='tight')


def main(modeldir):
    """
    Train neural posterior estimator.

    See also
    --------
    utils.setup_modeldir
    """
    modeldir = Path(modeldir)
    datadir = modeldir.resolve().parent/utils.TRAINING_DIR
    config = utils.load_model_config(modeldir)

    mask = np.load(datadir/utils.MASK_FILENAME)

    simulation_parameters = np.load(
        datadir/utils.FOLDED_SAMPLED_PARAMS_FILENAME
        )[mask][:config.MAX_TRAINING_EXAMPLES]

    simulation_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME
                             )[mask][:config.MAX_TRAINING_EXAMPLES]

    theta = torch.tensor(simulation_parameters, dtype=torch.float32
                        ).to(config.DEVICE)
    x = torch.tensor(simulation_data, dtype=torch.float32).to(config.DEVICE)

    neural_posterior = sbi.utils.posterior_nn(**config.POSTERIOR_NN_KWARGS)

    inference = SNPE(density_estimator=neural_posterior,
                     device=config.DEVICE,
                     summary_writer=SummaryWriter(modeldir)
                     ).append_simulations(theta, x)

    with Profile() as profiler:
        density_estimator = inference.train(**config.TRAIN_KWARGS)

    profiler.dump_stats(modeldir/'profiling')

    posterior = inference.build_posterior(density_estimator)
    torch.save(posterior, modeldir/utils.POSTERIOR_FILENAME)
    plot_logprob(modeldir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train neural posterior estimator from existing data')
    parser.add_argument('modeldir',
                        help='''Path of the run directory, must contain a
                                (populated) `training_data/` subdirectory.''')

    main(**vars(parser.parse_args()))
