import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from sbi.inference import SNPE
import sbi.utils

import cogwheel.utils

def main(sim_dir):
    """Training"""
    sim_dir = Path(sim_dir)

    simulation_parameters = np.load(
        sim_dir/'folded_sampled_params.npy')

    simulation_data = np.load(sim_dir/'compressed_data.npy')

    theta = torch.as_tensor(simulation_parameters, dtype=torch.float32)
    x = torch.as_tensor(simulation_data, dtype=torch.float32)

    neural_posterior = sbi.utils.posterior_nn(model="nsf", hidden_features=256)

    rundir = cogwheel.utils.get_rundir(sim_dir)

    inference = SNPE(
        density_estimator=neural_posterior, device='cuda',
        summary_writer=SummaryWriter(rundir))

    inference = inference.append_simulations(theta, x)

    density_estimator = inference.train(
        training_batch_size=4096, stop_after_epochs=50,
        learning_rate=0.001, show_train_summary=True)

    posterior = inference.build_posterior(density_estimator)

    torch.save(posterior, sim_dir/f'posterior_{rundir.name}.pt')

    print(posterior)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training NN from existing data')
    parser.add_argument(
        'sim_dir',
        help='''Training directory path, must contain files
                `folded_sampled_params.npy`. and `simulation_data.npy`.''')

    main(**vars(parser.parse_args()))
