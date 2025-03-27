"""Modifications to the behavior of ``sbi``."""
import numpy as np

import torch.utils.data
import sbi.inference


class NPEFixedBatches(sbi.inference.NPE):
    """
    Like sbi.inference.NPE except the batches are fixed.

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
        if dataloader_kwargs:
            print(f'Ignoring `{dataloader_kwargs=}`')

        dataset = torch.utils.data.TensorDataset(
            *self.get_simulations(starting_round))

        if not resume_training:
            # These allow to preserve the exact partition into batches
            # as well as training/validation over multiple trainings.
            self._train_ind_batches, self._val_ind_batches \
                = get_train_val_batch_inds(len(dataset), training_batch_size,
                                           validation_fraction)

            # Other methods assume this attribute exists
            self.train_indices = np.concatenate(self._train_ind_batches)

        train_batches = [dataset[inds] for inds in self._train_ind_batches]
        val_batches = [dataset[inds] for inds in self._val_ind_batches]

        train_loader = FixedBatchesDataLoader(train_batches)
        val_loader = FixedBatchesDataLoader(val_batches)

        return train_loader, val_loader


def get_train_val_batch_inds(num_simulations, training_batch_size,
                             validation_fraction):
    """
    Returns
    -------
    train_ind_batches, val_ind_batches : list of int arrays
        Indices of the training and validation simulations, arranged in
        batches.
    """
    num_batches = num_simulations // training_batch_size
    batch_indices = np.split(np.arange(num_batches * training_batch_size),
                             num_batches)

    num_training_batches = int(
        len(batch_indices) * (1-validation_fraction))
    train_ind_batches = batch_indices[:num_training_batches]
    val_ind_batches = batch_indices[num_training_batches:]
    return train_ind_batches, val_ind_batches


class FixedBatchesDataLoader:
    """A list of batches, always the same."""
    def __init__(self, batches, shuffle_batches=True):
        """
        Parameters
        ----------
        batches : list of lists of torch.Tensor
            Each batch contains multiple tensors, e.g. data and
            parameters.

        shuffle_batches : bool
            Whether to iterate over the batches in random order every
            time.
        """
        if len(set(map(len, batches))) != 1:
            raise ValueError('Batches are not the same size.')

        self.batches = batches
        self.shuffle_batches = shuffle_batches

        self.batch_size = len(batches[0][0])
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        order = np.arange(len(self.batches))
        if self.shuffle_batches:
            self._rng.shuffle(order)

        for i in order:
            yield self.batches[i]
