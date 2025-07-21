"""
Probabilistic unfolding of folded posterior samples.

"Unfolding" is a postprocessing step to undo the "folding", a technique
to mitigate known multimodalities in the posterior.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
import xgboost

from . import utils


class UnfoldingClassifier:
    """
    Predict the region onto which samples should be unfolded.

    The prediction is probabilistic, based on the data and the folded
    parameter values, using an XGBoost classifier.
    """
    def __init__(self, unfolderdir):
        self.unfolderdir = Path(unfolderdir)
        rundir = self.unfolderdir.resolve().parents[2]
        data_config = utils.load_data_config(rundir)

        # Setup booster
        self.config = utils.load_unfolder_config(self.unfolderdir)
        self.config.UNFOLDER_KWARGS['num_class'] \
            = 2 ** len(data_config.TRANSFORM_CLASS.folded_params)
        self.config.UNFOLDER_KWARGS['objective'] = 'multi:softprob'
        self.booster = xgboost.XGBClassifier(**self.config.UNFOLDER_KWARGS)

        # Load or train booster
        filename = self.unfolderdir/utils.UNFOLDER_FILENAME
        if filename.exists():
            self.booster.load_model(filename)
        else:  # Model has not been previously trained
            self.train()

    def train(self):
        """Train the XGBoost model and save it in `unfolderdir`."""
        # Load training data
        compressed_data, rescaled_params, unfolding_labels, weights \
            = self._load_data(self.unfolderdir, use_test_data=False)
        data = np.hstack([compressed_data, rescaled_params])

        # Fit
        print('Training XGBoost model...')
        self.booster.fit(data, unfolding_labels, sample_weight=weights)
        print('Done.')
        self.booster.save_model(self.unfolderdir/utils.UNFOLDER_FILENAME)

    def predict(self, compressed_data, rescaled_params):
        """Make predictions from the trained XGBoost model."""
        # Accept same `compressed_data` for many `rescaled_params`:
        compressed_data = np.broadcast_to(
            compressed_data,
            (rescaled_params.shape[0], compressed_data.shape[-1]))

        data = np.hstack([compressed_data, rescaled_params])

        return self.booster.predict_proba(data)

    def plot_confusion_matrix(self):
        """Plot confusion matrices for training and test sets."""
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(7.5, 4))
        axs[0].imshow(self.confusion_matrix(use_test_data=False))  # Train
        axs[1].imshow(self.confusion_matrix(use_test_data=True))  # Test

        fig.supxlabel('Predictions')
        fig.supylabel('True region')
        axs[0].set_title('Training set')
        axs[1].set_title('Test set')
        for ax in axs:
            for axis in ax.xaxis, ax.yaxis:
                axis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

    def confusion_matrix(self, use_test_data=True):
        """
        Compute the confusion matrix using the training or test data.

        Note that a diagonal confusion matrix is not the correct answer;
        if the folding symmetries were exact the confusion matrix would
        be uniform. In reality it should be something in between.

        Parameters
        ----------
        use_test_data : bool
            If True, the confusion matrix will be computed using the
            test data. If False, using the training data.
        """
        compressed_data, rescaled_params, unfolding_labels, weights = \
            self._load_data(self.unfolderdir, use_test_data)

        predictions = self.predict(compressed_data, rescaled_params)
        return self._confusion_matrix(predictions, unfolding_labels, weights)

    def _confusion_matrix(self, predictions, true_labels, weights):
        """
        Compute the confusion matrix for the given predictions.

        Parameters
        ----------
        predictions : (n_samples, 2**n_folded_params) float array
            Probabilities of unfolding into the different regions as
            predicted by the classifier. E.g. output of `.predict()`.

        true_labels : (n_samples,) int array
            Index of the region where each injection was actually made.
        """
        num_class = self.config.UNFOLDER_KWARGS['num_class']
        if predictions.shape[1] != num_class:
            raise ValueError(f'Expected predictions for {num_class} classes, '
                             f'got {predictions.shape[1]}.')

        confusion_matrix = np.zeros((num_class, num_class))

        # Populate the probabilistic confusion matrix
        for true_label, probabilities, weight in zip(
                true_labels, predictions, weights):
            confusion_matrix[true_label] += weight * probabilities

        # Normalize rows
        confusion_matrix /= confusion_matrix.sum(axis=1, keepdims=True) + 1e-9

        return confusion_matrix

    def _load_data(self, unfolderdir, use_test_data: bool):
        if use_test_data:
            foldername = utils.TEST_DIR
        else:
            foldername = utils.TRAINING_DIR

        rescalerdir, priordir, rundir = unfolderdir.resolve().parents[:3]
        datadir = rundir/foldername
        rescaled_params = np.load(
            rescalerdir/foldername/utils.RESCALED_PARAMETERS_FILENAME)
        mask = np.load(datadir/utils.MASK_FILENAME)
        compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]

        with h5py.File(datadir/utils.UNFOLDING_LABELS_FILENAME) as file:
            unfolding_labels = file['dataset'][:][mask]  # [:] makes it faster

        weights = np.load(priordir/foldername/utils.WEIGHTS_FILENAME)

        return compressed_data, rescaled_params, unfolding_labels, weights


def main(unfolderdir):
    """Train and save an UnfoldingClassifier if it doesn't exist."""
    UnfoldingClassifier(unfolderdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a model to predict unfolding probabilities.')
    parser.add_argument('unfolderdir', help='Run directory.')

    main(**vars(parser.parse_args()))
