"""Functions for training neural posterior estimators."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sbi.inference import SNPE
import sbi.utils

import cogwheel.utils

from . import utils


def main(sim_dir):
    """
    Train neural posterior estimator.

    This will create a `rundir` inside `sim_dir` with the trained
    posterior and training diagnostics.
    """
    sim_dir = Path(sim_dir)
    config = utils.load_config(sim_dir)

    mask = np.load(sim_dir/'mask.npy')
    simulation_parameters = np.load(sim_dir/'folded_sampled_params.npy'
                                   )[mask][:config.MAX_TRAINING_EXAMPLES]
    simulation_data = np.load(sim_dir/'compressed_data.npy'
                             )[mask][:config.MAX_TRAINING_EXAMPLES]

    theta = torch.as_tensor(simulation_parameters, dtype=torch.float32)
    x = torch.as_tensor(simulation_data, dtype=torch.float32)

    neural_posterior = sbi.utils.posterior_nn(**config.POSTERIOR_NN_KWARGS)

    rundir = cogwheel.utils.get_rundir(sim_dir)

    inference = SNPE(density_estimator=neural_posterior,
                     device=config.DEVICE,
                     summary_writer=SummaryWriter(rundir)
                     ).append_simulations(theta, x)

    density_estimator = inference.train(**config.TRAIN_KWARGS)
    posterior = inference.build_posterior(density_estimator)
    torch.save(posterior, rundir/'posterior.pt')
    print(posterior)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training NN from existing data')
    parser.add_argument(
        'sim_dir',
        help='''Training directory path, must contain files
                `folded_sampled_params.npy`. and `simulation_data.npy`.''')

    main(**vars(parser.parse_args()))
