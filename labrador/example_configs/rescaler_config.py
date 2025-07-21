"""
Example configuration file with parameters for the multilayer perceptron
that learns mean and covariance of the posterior, to rescale the
parameters before passing them to sbi

This file may be copied into ``rescalerdir`` and edited before training
the network.
"""
import torch

# kwargs for the multilayer perceptron
RESCALER_NN_KWARGS = {'n_layers': 5,
                      'layer_size': 100,
                      'activation_fn': 'SiLU'}

RESCALER_TRAIN_KWARGS = {
    'training_batch_size': 8192,
    'validation_fraction': 0.1,
    'stop_after_epochs': 200,
    'max_num_epochs': 10000,
    'optimizer_cls': torch.optim.AdamW,
    'optimizer_kwargs': {'lr': 1e-4},  # kwargs to `optimizer_cls`
    'scheduler_kwargs': {
        'factor': 0.5,
        'patience': 64,
        'min_lr': 5e-6,
     },  # kwargs to torch.optim.lr_scheduler.ReduceLROnPlateau
}

COMPACTIFICATION = 'gaussian'  # 'gaussian' or 'tanh'

DEVICE = None  # ``None`` will try to use 'cuda' or fall back to 'cpu'.
