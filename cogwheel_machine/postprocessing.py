"""
UnfoldingClassifier
    Inputs
        compressed_data
        rescaled_parameters
    Outputs
        Unfolding probabilities

ParameterRescaler
    Inputs
        compressed_data
        rescaled_parameters
    Outputs
        folded_sampled_parameters

PostProcessor._unfold
    Inputs
        Unfolding probabilities
        folded_sampled_parameters
    Outputs
        sampled_parameters

Transform
    Inputs
        transform_kwargs
        sampled_parameters
    Outputs
        standard_params
"""
import numpy as np
import pandas as pd
import torch

_searchsorted_v = np.vectorize(np.searchsorted,
                               signature='(n),()->()', otypes=[int])


class PostProcessor:
    """
    Turn folded rescaled parameters into unfolded standard parameters.

    This class orchestrates the different transformations that need to
    take place to turn the output of the normalizing flow (samples of
    folded rescaled parameters) into usable physical parameters.
    """

    def __init__(self,
                 unfolding_classifier,
                 parameter_rescaler,
                 transform):
        """
        Parameters
        ----------
        unfolding_classifier : unfolding.UnfoldingClassifier
            Predicts the probabilities of unfolding into each of the
            2**len(transform.folded_params) regions of parameter space.

        parameter_rescaler : rescaling.ParameterRescaler
            Implements a data-dependent coordinate transformation that
            makes the posterior approximately standard normal.

        transform : transform.TransformMixin
            Transforms between standard parameters and coordinates
            suitable for folding.
        """
        self.unfolding_classifier = unfolding_classifier
        self.parameter_rescaler = parameter_rescaler
        self.transform = transform
        self._unfold_v = np.vectorize(self.transform.unfold,
                                      signature='(n)->(m,n)', otypes=[float])

    def postprocess_samples(self, compressed_data, rescaled_parameters):
        """End to end, from folded-rescaled to standard parameters."""
        with torch.no_grad():
            folded_sampled_parameters = self.parameter_rescaler.unrescale(
                compressed_data, rescaled_parameters)

        unfolding_probabilities = self.unfolding_classifier.predict(
            compressed_data, rescaled_parameters)

        sampled_parameters = self._unfold(unfolding_probabilities,
                                          folded_sampled_parameters)

        self.transform.transform_samples(sampled_parameters)

        return sampled_parameters

    def _unfold(self, unfolding_probabilities,
                folded_sampled_parameters):
        """
        Choose a random unfolding of the folded parameters (vectorized).

        Parameters
        ----------
        unfolding_probabilities : (n_samples, n_unfolding) float array
            For each sample, probability of unfolding into each of the
            2**n_folded_params regions of parameter space.
            Should sum to 1 along the ``n_unfolding`` axis.

        folded_sampled_parameters : (n_samples, n_sampled_params) array
            Folded parameters.

        Returns
        -------
        unfolded_sampled_parameters : pandas.DataFrame
            Sampled parameters in the unfolded space. It has shape
            (n_samples, n_sampled_parameters).
        """
        unfolded = self._unfold_v(folded_sampled_parameters)
        cumprobs = np.cumsum(unfolding_probabilities, axis=-1)

        rng = np.random.default_rng()
        inds = _searchsorted_v(cumprobs, rng.uniform(size=len(cumprobs)))

        return pd.DataFrame(unfolded[np.arange(len(inds)), inds],
                            columns=self.transform.sampled_params)
