"""
Example configuration file with parameters for the multilayer perceptron
that learns mean and covariance of the posterior, to rescale the
parameters before passing them to sbi

This file may be copied into ``rescalerdir`` and edited before training
the network.
"""
import torch

# kwargs for the multilayer perceptron
RESCALER_NN_KWARGS = {'n_layers': 4,
                      'layer_size': 32,
                      'activation_fn': 'SiLU'}

RESCALER_TRAIN_KWARGS = {
    'training_batch_size': 4000,
    'validation_fraction': 0.1,
    'stop_after_epochs': 32,
    'max_num_epochs': 300,
    'optimizer_cls': torch.optim.AdamW,
    'optimizer_kwargs': {'lr': 5e-3},  # kwargs to `optimizer_cls`
    'scheduler_kwargs': {
        'factor': 0.5,
        'patience': 8,
        'min_lr': 1e-5,
     },  # kwargs to torch.optim.lr_scheduler.ReduceLROnPlateau
}

DEVICE = None  # ``None`` will try to use 'cuda' or fall back to 'cpu'.
