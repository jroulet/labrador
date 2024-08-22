"""Functions for training neural posterior estimators."""
import argparse
from pathlib import Path
from cProfile import Profile
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn 
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
        

class SummaryNet(nn.Module): 
    
    def __init__(self): 
        super().__init__()
        # 1D convolutional layer
        #self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # Maxpool layer that reduces size
        #self.pool = nn.MaxPool1d(kernel_size=8, stride=8)
        # Fully connected layer
        self.fc = nn.Linear(in_features=in_dim, out_features=out_dim) 
        
    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = x.view(-1, 6*4*4)
        x = F.relu(self.fc(x))
        return x


class FullyConnectedEmbeddingNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes):
        """
        Initializes the FullyConnectedEmbeddingNetwork.

        Parameters:
        - input_size (int): The size of the input features.
        - layer_sizes (list of int): A list where each element
        is the size of the corresponding hidden layer.
        """
        super(FullyConnectedEmbeddingNetwork, self).__init__()

        # Create a list of fully connected layers
        layers = []
        in_size = input_size
        for i, size in enumerate(layer_sizes):
            layers.append(nn.Linear(in_size, size))
            if i < len(layer_sizes) - 1:
                layers.append(nn.ReLU())  # Add ReLU activation only after hidden layers
            in_size = size

        # Combine the layers into a sequential model
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
        - torch.Tensor: Output tensor of shape (batch_size, final_layer_size).
        """
        return self.fc_layers(x)


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

    if config.EMBEDDING_LAYER_SIZES:
        embedding_net = FullyConnectedEmbeddingNetwork(
            input_size=x.shape[1], layer_sizes=config.EMBEDDING_LAYER_SIZES)
    else:
        embedding_net = None

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
