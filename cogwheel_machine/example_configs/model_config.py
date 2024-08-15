"""
Example configuration file with parameters for the neural posterior
estimation.

This file may be copied into `run_dir` and edited before training the
network.
"""
# kwargs to sbi.utils.posterior_nn
POSTERIOR_NN_KWARGS = {'model': 'nsf',
                       'hidden_features': 64}

# kwargs to sbi.inference.SNPE.train
TRAIN_KWARGS = {'training_batch_size': 8192,
                'stop_after_epochs': 32,
                'learning_rate': 1e-3,
                'show_train_summary': True}

MAX_TRAINING_EXAMPLES = None  # int
DEVICE = 'cuda'
