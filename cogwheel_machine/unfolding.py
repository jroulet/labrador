"""
Classes for probabilistic unfolding of folded posterior samples.

"Unfolding" being a postprocessing step to undo the "folding", a
technique to mitigate known multimodalities in the posterior.
"""
from pathlib import Path
import numpy as np
import h5py
import xgboost
from . import utils


class UnfoldingClassifier:
    """
    Predict the region onto which samples should be unfolded.

    The prediction is probabilistic, based on the data and the folded
    parameter values, using an XGBoost classifier.
    """
    def __init__(self, rundir):
        self.rundir = Path(rundir)

        self.config = utils.load_data_config(self.rundir)
        self.booster = xgboost.XGBClassifier(**self.config.UNFOLDER_KWARGS)
        filename = self.rundir/utils.UNFOLDER_FILENAME
        if filename.exists():
            self.booster.load_model(filename)
        else:  # Model has not been previously trained
            self.train()

    def train(self):
        """
        Train the XGBoost model and save it in `.rundir`.
        """
        # Load training data
        datadir = self.rundir/utils.TRAINING_DIR
        compressed_data, rescaled_params, unfolding_labels = self._load_data(
            datadir)
        data = np.hstack([compressed_data, rescaled_params])

        # Fit
        print('Training XGBoost model...')
        self.booster.fit(data, unfolding_labels)
        print('Done.')
        self.booster.save_model(self.rundir/utils.UNFOLDER_FILENAME)

    def predict(self, compressed_data, rescaled_params):
        """
        Make predictions from the trained XGBoost model.
        """
        data = np.hstack([compressed_data, rescaled_params])
        return self.booster.predict_proba(data)

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
        if use_test_data:
            datadir = self.rundir/utils.TEST_DIR
        else:
            datadir = self.rundir/utils.TRAINING_DIR
        compressed_data, rescaled_params, unfolding_labels = self._load_data(
            datadir)

        predictions = self.predict(compressed_data, rescaled_params)
        return self._confusion_matrix(predictions, unfolding_labels)

    def _confusion_matrix(self, predictions, true_labels):
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
        for true_label, probabilities in zip(true_labels, predictions):
            confusion_matrix[true_label] += probabilities

        # Normalize rows
        confusion_matrix /= confusion_matrix.sum(axis=1, keepdims=True) + 1e-9

        return confusion_matrix

    def _load_data(self, datadir):
        rescaled_params = np.load(
            datadir/utils.RESCALED_PARAMETERS_FILENAME)
        mask = np.load(datadir/utils.MASK_FILENAME)
        compressed_data = np.load(datadir/utils.COMPRESSED_DATA_FILENAME)[mask]

        with h5py.File(datadir/utils.UNFOLDING_LABELS_FILENAME) as file:
            unfolding_labels = file['dataset'][mask]

        return compressed_data, rescaled_params, unfolding_labels
