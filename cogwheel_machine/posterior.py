"""
Generate samples and their probability density in standard coordinates.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import cogwheel.prior
import cogwheel.utils

from cogwheel_machine import rescaling, training, unfolding, utils


class Posterior:
    """
    Generate samples in the space of physical parameters.

    This class orchestrates the different transformations that need to
    take place to turn the output of the normalizing flow (samples of
    folded rescaled parameters) into usable physical parameters.
    """

    @classmethod
    def from_tree(cls, sbidir, unfolderdir):
        """
        Parameters
        ----------
        sbidir : os.PathLike
            Path to the SBI directory, lives inside ``rescalerdir``.

        unfolderdir : os.PathLike
            Path to the unfolder directory, also lives inside
            ``rescalerdir``.

        Raises
        ------
        ValueError
            If `sbidir` and `unfolderdir` don't have the same parent.
        """
        sbidir = Path(sbidir).resolve()
        unfolderdir = Path(unfolderdir).resolve()
        rescalerdir, _, rundir = sbidir.parents[:3]

        if unfolderdir.parent != rescalerdir:
            raise ValueError(
                '`sbidir` and `unfolderdir` are not in the same `rescalerdir`')

        sbi_posterior = training.load_posterior(sbidir)
        unfolding_classifier = unfolding.UnfoldingClassifier(unfolderdir)
        parameter_rescaler = rescaling.ParameterRescaler(rescalerdir)

        data_config = utils.load_data_config(rundir)
        simulation_prior = data_config.PRIOR_CLASS(**data_config.PRIOR_KWARGS)
        fixed_par_dic = _get_fixed_par_dic(simulation_prior)
        return cls(parameter_rescaler=parameter_rescaler,
                   sbi_posterior=sbi_posterior,
                   unfolding_classifier=unfolding_classifier,
                   fixed_par_dic=fixed_par_dic)

    def __init__(self,
                 parameter_rescaler,
                 sbi_posterior,
                 unfolding_classifier,
                 fixed_par_dic=None,
                 ):
        """
        Parameters
        ----------
        parameter_rescaler : rescaling.ParameterRescaler
            Implements a data-dependent coordinate transformation that
            makes the posterior approximately standard normal.

        sbi_posterior : sbi.inference.DirectPosterior
            Predicts the posterior over folded-rescaled parameters using
            a conditional normalizing flow.

        unfolding_classifier : unfolding.UnfoldingClassifier
            Predicts the probabilities of unfolding into each of the
            2**len(transform.folded_params) regions of parameter space.

        fixed_par_dic : dict
            Contains parameter values that are fixed in all the samples
            (e.g. reference frequency, tidal deformabilities, ...)

        See Also
        --------
        from_tree : To load these inputs from the directory tree.
        """
        self.parameter_rescaler = parameter_rescaler
        self.sbi_posterior = sbi_posterior
        self.unfolding_classifier = unfolding_classifier
        self.fixed_par_dic = fixed_par_dic or {}

    def generate_samples_and_lnprob(self, n_samples, compressed_data,
                                    transform):
        """
        Generate samples and their probablilty density in the space
        of standard parameters.

        Parameters
        ----------
        n_samples : int
            How many samples to generate

        compressed_data : (n_samples, n_compressed_params) array
            Compressed data, i.e. the output of the compression step.
            This is the input to the normalizing flow.

        transform : transform.TransformMixin
            Transforms between standard parameters and coordinates
            suitable for folding.

        Returns
        -------
        samples : pd.DataFrame
            Columns contain `transform.standard_params` and
            `transform.sampled_params`

        lnprob_standard : float array
            Log probability density in the space of standard parameters
            (`transform.standard_params`).
            It is supposed to resemble the log posterior to the extent
            that the model is well trained, but it is guaranteed to
            describe the distribution of the samples including the
            normalization.
        """
        rescaled_parameters = self.sbi_posterior.sample(
            [n_samples], x=compressed_data, show_progress_bars=False)

        lnp_sbi = self.sbi_posterior.log_prob(rescaled_parameters,
                                              x=compressed_data).numpy()

        samples, lnj = self.unrescale_unfold_transform(
            compressed_data, transform, rescaled_parameters)

        cogwheel.utils.update_dataframe(samples, self.fixed_par_dic)
        # lnj := log |∂{standard} / ∂{folded_rescaled}|
        # p(standard) = p(folded_rescaled) / |∂{standard}/∂{folded_rescaled}|
        return samples, lnp_sbi - lnj

    def unrescale_unfold_transform(self, compressed_data, transform,
                                   rescaled_parameters):
        """
        End to end, from folded-rescaled to standard parameters.

        Parameters
        ----------
        compressed_data : (n_samples, n_compressed_params) array
            Compressed data, i.e. the output of the compression step.
            This is the input to the normalizing flow.

        transform : transform.TransformMixin
            Transforms between standard parameters and coordinates
            suitable for folding.

        rescaled_parameters : (n_samples, n_rescaled_params) array
            Rescaled parameters, i.e. the output of the normalizing flow.
            This is the input to the postprocessor.

        Returns
        -------
        parameters : pandas.DataFrame
            Parameters in the physical space.

        lnj : (n_samples,) float array
            Logarithm of the Jacobian of the transformation from
            folded-rescaled to unfolded standard parameters.
            log |∂{standard} / ∂{folded_rescaled}|
        """
        # • Unrescale:
        with torch.no_grad():
            folded_sampled_parameters, lnj_unrescale \
                = self.parameter_rescaler.unrescale(compressed_data,
                                                    rescaled_parameters)
        folded_sampled_parameters = folded_sampled_parameters.cpu()
        # log |∂{folded} / ∂{folded_rescaled}|
        lnj_unrescale = lnj_unrescale.cpu().numpy()

        # • Unfold:
        unfolding_probabilities = self.unfolding_classifier.predict(
            compressed_data, rescaled_parameters)
        parameters, lnp_unfold = _unfold(transform,
                                         unfolding_probabilities,
                                         folded_sampled_parameters)
        # unfolded = (folded, unfold)
        # => p(unfolded) = p(folded) p(unfold | folded)
        # p(unfold | folded) plays the role of the "Jacobian":
        # p(unfold | folded) = |∂{folded} / ∂{unfolded}|

        # • Transform to standard coordinates:
        transform.transform_samples(parameters)
        lnj_v = np.vectorize(transform.ln_jacobian_determinant, otypes=[float])
        # log |∂{unfolded} / ∂{standard}|
        lnj_inverse_transform = lnj_v(**parameters[transform.standard_params])

        # Obtain the total Jacobian as the product
        # |∂{standard} / ∂{folded_rescaled}| = (
        #     |∂{folded} / ∂{folded_rescaled}|
        #     |∂{unfolded} / ∂{folded}|
        #     |∂{standard} / ∂{unfolded}|
        # )
        lnj = lnj_unrescale - lnp_unfold - lnj_inverse_transform
        # Add extra information for debugging purposes (TODO remove?)
        rescaled_df = pd.DataFrame(
            rescaled_parameters,
            columns=[f'rescaled_{par}' for par in transform.sampled_params])
        cogwheel.utils.update_dataframe(parameters, rescaled_df)
        parameters['lnj_unrescale'] = lnj_unrescale
        parameters['lnp_unfold'] = lnp_unfold
        parameters['lnj_inverse_transform'] = lnj_inverse_transform
        parameters['lnj'] = lnj

        return parameters, lnj

    def inversetransform_fold_rescale(self, compressed_data, transform,
                                      samples):
        """
        From physical parameters to folded-rescaled parameters.

        Inverse of ``.unrescale_unfold_transform()`` (but note that
        ``.unrescale_unfold_transform()`` is not the inverse of this
        function because folding is not invertible).

        Parameters
        ----------
        compressed_data : (n_samples, n_compressed_params) array
            Compressed data, i.e. the output of the compression step.
            This is the input to the normalizing flow.

        transform : transform.TransformMixin
            Transforms between standard parameters and coordinates
            suitable for folding.

        samples : pandas.DataFrame
            Parameters in the physical space.

        Returns
        -------
        rescaled_parameters : (n_samples, n_rescaled_params) array
            Rescaled-folded parameters.
        """
        # Inverse-transform to coordinates suitable for folding:
        samples = samples.copy()  # Leave input untouched
        transform.inverse_transform_samples(samples)

        # Fold:
        folded = np.vectorize(
            transform.fold,
            signature=','.join('()' for _ in transform.sampled_params)+'->(n)'
        )(**samples[transform.sampled_params])

        # Rescale:
        rescaled = self.parameter_rescaler.rescale(compressed_data, folded)
        return rescaled

def _unfold(transform, unfolding_probabilities,
            folded_sampled_parameters):
    """
    Choose a random unfolding of the folded parameters (vectorized).

    Parameters
    ----------
    transform : transform.TransformMixin
        Transforms between standard parameters and coordinates suitable
        for folding.

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

    lnp_unfolding : (n_samples,) float array
        Logarithm of the unfolding probability.
    """
    unfold_v = np.vectorize(transform.unfold,
                            signature='(n)->(m,n)', otypes=[float])
    unfolded = unfold_v(folded_sampled_parameters)
    cumprobs = np.cumsum(unfolding_probabilities, axis=-1)
    np.testing.assert_allclose(cumprobs[..., -1], 1, rtol=1e-5)
    n_samples = len(cumprobs)

    rng = np.random.default_rng()
    inds = (np.arange(n_samples),
            _searchsorted_v(cumprobs, rng.uniform(size=n_samples)))

    unfolded_sampled_parameters = pd.DataFrame(
        unfolded[inds], columns=transform.sampled_params)
    lnp_unfolding = np.log(unfolding_probabilities[inds])

    return unfolded_sampled_parameters, lnp_unfolding


_searchsorted_v = np.vectorize(np.searchsorted,
                               signature='(n),()->()', otypes=[int])


def _get_fixed_par_dic(prior):
    """
    Return dictionary with all fixed parameters in a prior.

    Parameters
    ---------
    prior : cogwheel.prior.Prior
    """
    return cogwheel.utils.merge_dictionaries_safely(*_fixed_par_dics(prior))


def _fixed_par_dics(prior):
    """
    Generator that yields dictionaries with fixed parameters in a prior.

    Parameters
    ---------
    prior : cogwheel.prior.Prior
    """
    if isinstance(prior, cogwheel.prior.FixedPrior):
        yield prior.standard_par_dic

    elif isinstance(prior, cogwheel.prior.CombinedPrior):
        for subprior in prior.subpriors:
            yield from _fixed_par_dics(subprior)
