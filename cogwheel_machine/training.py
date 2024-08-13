"""Functions for training neural posterior estimators."""
import argparse
from pathlib import Path
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

from sbi.inference import SNPE
import sbi.utils

from cogwheel_machine import utils


def main(modeldir):
    """
    Train neural posterior estimator.
    """
    modeldir = Path(modeldir)
    datadir = modeldir.parent/utils.TRAINING_DIR
    config = utils.load_model_config(modeldir)

    mask = np.load(datadir/utils.MASK_FILENAME)

    simulation_parameters = np.load(
        datadir/utils.FOLDED_SAMPLED_PARAMS_FILENAME
        )[mask][:config.MAX_TRAINING_EXAMPLES]

    simulation_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME
                             )[mask][:config.MAX_TRAINING_EXAMPLES]

    theta = torch.as_tensor(simulation_parameters, dtype=torch.float32)
    x = torch.as_tensor(simulation_data, dtype=torch.float32)

    neural_posterior = sbi.utils.posterior_nn(**config.POSTERIOR_NN_KWARGS)

    inference = SNPE(density_estimator=neural_posterior,
                     device=config.DEVICE,
                     summary_writer=SummaryWriter(modeldir)
                     ).append_simulations(theta, x)

    density_estimator = inference.train(**config.TRAIN_KWARGS)
    posterior = inference.build_posterior(density_estimator)
    torch.save(posterior, modeldir/utils.POSTERIOR_FILENAME)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train neural posterior estimator from existing data')
    parser.add_argument('rundir',
                        help='''Path of the run directory, must contain a
                                (populated) `training_data/` subdirectory.''')

    main(**vars(parser.parse_args()))
