"""Modifications to the behavior of ``sbi``."""
import numpy as np

import torch.utils.data
import sbi.inference


class SNPEFixedBatches(sbi.inference.SNPE):
    """
    Like sbi.inference.SNPE except the batches are fixed.

    The batches are made of consecutive simulations (no shuffling), the
    first batches are training and the last are validation.

    By making the simulations consecutive we try to preserve the low-
    discrepancy property of QMC sequences.
    By making the batches once and for all we speed up iterations over
    the data.
    """
    def get_dataloaders(self,
                        starting_round: int = 0,
                        training_batch_size: int = 200,
                        validation_fraction: float = 0.1,
                        resume_training=None,
                        dataloader_kwargs=None):
        """Return dataloaders for training and validation."""
        if resume_training:
            raise NotImplementedError
        if dataloader_kwargs:
            print(f'Ignoring `{dataloader_kwargs=}`')

        dataset = torch.utils.data.TensorDataset(
            *self.get_simulations(starting_round))

        num_batches = len(dataset) // training_batch_size
        batch_indices = np.split(np.arange(num_batches * training_batch_size),
                                 num_batches)
        batches = [dataset[inds] for inds in batch_indices]

        num_training_batches = int(len(batches) * (1-validation_fraction))

        self.train_indices = np.concatenate(
            batch_indices[:num_training_batches])
        self.val_indices = np.concatenate(
            batch_indices[num_training_batches:])

        train_loader = FixedBatchesDataLoader(batches[:num_training_batches])
        val_loader = FixedBatchesDataLoader(batches[num_training_batches:])

        return train_loader, val_loader


class FixedBatchesDataLoader(list):
    """A list of batches, always the same."""
    def __init__(self, batches):
        if len(set(map(len, batches))) != 1:
            raise ValueError('Batches are not the same size.')

        super().__init__(batches)
        self.batch_size = len(batches[0][0])
