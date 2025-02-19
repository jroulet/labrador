"""
UnfoldingClassifier
    Inputs
        compressed_data
        rescaled_params
    Outputs
        Unfolding probabilities

ParameterRescaler
    Inputs
        compressed_data
        rescaled_params
    Outputs
        folded_sampled_params

PostProcessor._unfold
    Inputs
        Unfolding probabilities
        folded_sampled_params
    Outputs
        sampled_params

Transform
    Inputs
        transform_kwargs
        sampled_params
    Outputs
        standard_params
"""
import numpy as np
import pandas as pd
import torch


@np.vectorize
def _flip_binary_number(i):
    # Represent the number as a 4-digit binary string
    binary_str = format(i, '04b')
    # Reverse the string (flip the digits)
    flipped_str = binary_str[::-1]
    # Convert the flipped binary string back to an integer
    flipped_num = int(flipped_str, 2)
    return flipped_num


class PostProcessor:
    """
    Turn folded rescaled parameters into unfolded standard parameters.

    This class orchestrates the different transformations that need to
    take place to turn the output of the normalizing flow (samples of
    folded rescaled parameters) into usable physical parameters.
    """
    _FLIPPED_NUMBERS = _flip_binary_number(np.arange(16))

    def __init__(self,
                 unfolding_classifier,
                 parameter_rescaler,
                 transform):
        """
        Parameters
        ----------
        unfolding_classifier: unfolding.UnfoldingClassifier
            Predicts the probabilities of unfolding into each of the
            2**len(transform.folded_params) regions of parameter space.

        parameter_rescaler: rescaling.ParameterRescaler
            Implements a data-dependent coordinate transformation that
            makes the posterior approximately standard normal.

        transform: transform.TransformMixin
            Transforms between standard parameters and coordinates
            suitable for folding.
        """
        self.unfolding_classifier = unfolding_classifier
        self.parameter_rescaler = parameter_rescaler
        self.transform = transform

    def postprocess_samples(self, compressed_data, rescaled_params):
        """
        End to end, from folded-rescaled to standard parameters.
        """
        with torch.no_grad():
            folded_sampled_params = self.parameter_rescaler.unrescale(
                compressed_data, rescaled_params)

        unfolding_probabilities = self.unfolding_classifier.predict(
            compressed_data, rescaled_params)

        sampled_params = self._unfold(unfolding_probabilities,
                                      folded_sampled_params)

        self.transform.transform_samples(sampled_params)

        return sampled_params

    def _unfold(self, unfolding_probabilities, folded_sampled_params):
        """
        Choose a random unfolding of the folded parameters, vectorized.

        Parameters
        ----------
        unfolding_probabilities : (n_samples, n_unfolding) float array
            For each sample, probability of unfolding into each of the
            2**n_folded_params regions of parameter space.
            Should sum to 1 along the ``n_unfolding`` axis.

        folded_sampled_params : (n_samples, n_sampled_params) array
            Folded parameters.

        Returns
        -------
        unfolded_sampled_params : pandas.DataFrame
            Sampled parameters in the unfolded space. It has shape
            (n_samples, n_sampled_params).
        """
        # Unfortunately the unfolding labels are not in the same order
        # as the transform.unfold.
        # For now, reorder the unfolding_probabilities to match the
        # transform.unfold.
        # TODO: redefine how the unfolding_labels are computed.
        unfolding_probabilities \
            = unfolding_probabilities[..., self._FLIPPED_NUMBERS]

        unfold = np.vectorize(self.transform.unfold, signature='(n)->(m,n)')
        unfolded = unfold(folded_sampled_params)

        cumprobs = np.cumsum(unfolding_probabilities, axis=-1)

        searchsorted = np.vectorize(np.searchsorted, signature='(n),()->()')
        rng = np.random.default_rng()
        inds = searchsorted(cumprobs,
                            rng.uniform(size=len(folded_sampled_params)))

        return pd.DataFrame(unfolded[np.arange(len(inds)), inds],
                            columns=self.transform.sampled_params)
