"""
Generate amortized samples, and reweight using likelihood evaluations.
"""
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

import cogwheel.prior
import cogwheel.skyloc_angles
import cogwheel.utils
import lal

from . import rescaling, training, unfolding, utils


class Posterior:
    """
    Generate samples in the space of physical parameters.

    This class orchestrates the different transformations that need to
    take place to turn the output of the normalizing flow (samples of
    folded rescaled parameters) into usable physical parameters.
    """

    @classmethod
    def from_tree(cls, tree):
        """
        Parameters
        ----------
        tree : utils.Tree
            Directory structure of the trained model.
        """
        sbi_posterior = training.load_posterior(tree.sbidir)
        unfolding_classifier = unfolding.UnfoldingClassifier(tree.unfolderdir)
        parameter_rescaler = rescaling.ParameterRescaler(tree.rescalerdir)
        data_config = utils.load_data_config(tree.rundir)
        simulation_prior = data_config.PRIOR_CLASS(**data_config.PRIOR_KWARGS)
        fixed_par_dic = _get_fixed_par_dic(simulation_prior)
        return cls(parameter_rescaler=parameter_rescaler,
                   sbi_posterior=sbi_posterior,
                   unfolding_classifier=unfolding_classifier,
                   fixed_par_dic=fixed_par_dic,
                   tgps_fiducial=data_config.TGPS)

    def __init__(self,
                 parameter_rescaler,
                 sbi_posterior,
                 unfolding_classifier,
                 fixed_par_dic=None,
                 tgps_fiducial=None,
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
            (e.g. reference frequency, tidal deformabilities, ...).

        tgps_fiducial : float
            GPS time with which the model has been trained.

        See Also
        --------
        from_tree : To load these inputs from the directory tree.
        """
        self.parameter_rescaler = parameter_rescaler
        self.sbi_posterior = sbi_posterior
        self.unfolding_classifier = unfolding_classifier
        self.fixed_par_dic = fixed_par_dic or {}
        self.tgps_fiducial = tgps_fiducial

    def generate_samples_and_lnprob(self, n_samples, compressed_data,
                                    transform, tgps_actual=None,
                                    dropna=True, verbose=True):
        """
        Generate samples and their probablilty density in the space of
        standard parameters.

        Parameters
        ----------
        n_samples : int
            How many samples to generate.

        compressed_data : (n_samples, n_compressed_params) array
            Compressed data, i.e. the output of the compression step.
            This is the input to the normalizing flow.

        transform : transform.TransformMixin
            Transforms between standard parameters and coordinates
            suitable for folding.

        tgps_actual : float
            Actual GPS time of the event, as opposed to the fiducial one
            with which the model is trained. Only needed if the
            posterior involves the sky location (right ascension).
            Hint: use ``event_data.tgps``.

        dropna : bool
            Discard SBI samples that produce unphysical parameters.
            (This may reduce the number of samples from `n_samples`.)

        verbose : bool
            Whether to report in case unphysical samples had to be
            dropped.

        Returns
        -------
        samples : pd.DataFrame
            Columns contain `transform.standard_params` and
            `transform.sampled_params`.

        standard_lnprob : float array
            Log probability density in the space of standard parameters
            (`transform.standard_params`).
            It is supposed to resemble the log posterior to the extent
            that the model is well trained, but it is guaranteed to
            describe the distribution of the samples (including the
            normalization).
        """
        rescaled_parameters = self.sbi_posterior.sample(
            [n_samples], x=compressed_data, show_progress_bars=False)

        lnp_sbi = self.sbi_posterior.log_prob(rescaled_parameters,
                                              x=compressed_data).numpy()

        samples, lnj = self.unrescale_unfold_transform(
            compressed_data, transform, rescaled_parameters, tgps_actual)

        cogwheel.utils.update_dataframe(samples, self.fixed_par_dic)

        if dropna:
            valid = samples.notna().all(axis=1)

            if verbose and not all(valid):
                print('Dropping unphysical samples '
                      f'({(~valid).mean() * 100:.3g}% of the total).')

            samples = samples[valid].reset_index(drop=True)
            lnp_sbi = lnp_sbi[valid]
            lnj = lnj[valid]

        # lnj := log |∂{standard} / ∂{rescaled}|
        # p(standard) = p(rescaled) / |∂{standard}/∂{rescaled}|
        return samples, lnp_sbi - lnj

    def unrescale_unfold_transform(self, compressed_data, transform,
                                   rescaled_parameters,
                                   tgps_actual=None):
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
            Rescaled parameters, i.e. output of the normalizing flow.
            This is the input to the postprocessor.

        tgps_actual : float
            Actual GPS time of the event, as opposed to the fiducial one
            with which the model is trained. Only needed if the
            posterior involves the sky location (right ascension).

        Returns
        -------
        parameters : pandas.DataFrame
            Parameters in the physical space.

        lnj : (n_samples,) float array
            Logarithm of the Jacobian of the transformation from
            folded-rescaled to unfolded standard parameters::

                log |∂{standard} / ∂{folded_rescaled}|
        """
        if 'ra' in transform.standard_params and tgps_actual is None:
            raise ValueError('Unknown `tgps_actual`.')

        # • Unrescale:
        with torch.no_grad():
            folded_sampled_parameters, lnj_unrescale \
                = self.parameter_rescaler.unrescale(compressed_data,
                                                    rescaled_parameters)
        folded_sampled_parameters = folded_sampled_parameters.cpu()
        # log |∂{folded} / ∂{rescaled}|
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
        # p(unfold | folded) ≡ |∂{folded} / ∂{unfolded}|

        # • Transform to standard coordinates:
        transform.transform_samples(parameters)
        lnj_v = np.vectorize(transform.ln_jacobian_determinant, otypes=[float])
        # log |∂{unfolded} / ∂{standard}|
        lnj_inverse_transform = lnj_v(**parameters[transform.standard_params])

        # Obtain the total Jacobian as the product
        # |∂{standard} / ∂{rescaled}| = (
        #     |∂{folded} / ∂{rescaled}|
        #     |∂{unfolded} / ∂{folded}|
        #     |∂{standard} / ∂{unfolded}|
        # )
        lnj = lnj_unrescale - lnp_unfold - lnj_inverse_transform
        # Add extra information for debugging purposes (TODO remove?)
        rescaled_df = pd.DataFrame(
            rescaled_parameters,
            columns=[f'rescaled_{par}' for par in transform.sampled_params])
        cogwheel.utils.update_dataframe(parameters, rescaled_df)

        # Correct RA
        if 'ra' in parameters:
            correct_right_ascension(parameters,
                                    tgps_actual=tgps_actual,
                                    tgps_fiducial=self.tgps_fiducial)

        return parameters, lnj

    def inversetransform_fold_rescale(self, compressed_data, transform,
                                      samples, tgps_actual=None):
        """
        From physical parameters to folded-rescaled parameters.

        Inverse of :py:meth:`unrescale_unfold_transform` (but note that
        :py:meth:`unrescale_unfold_transform` is not the inverse of this
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

        tgps_actual : float
            Actual GPS time of the event, as opposed to the fiducial one
            with which the model is trained. Only needed if the
            posterior involves the sky location (right ascension).

        Returns
        -------
        rescaled_parameters : (n_samples, n_rescaled_params) tensor
            Rescaled-folded parameters.
        """
        samples = samples.copy()  # Leave input untouched

        # Anti-correct RA
        if 'ra' in samples:
            if tgps_actual is None:
                raise ValueError('Unknown `tgps_actual`.')
            del samples['lon']
        correct_right_ascension(
            samples,
            tgps_actual=self.tgps_fiducial,  # Note opposite order
            tgps_fiducial=tgps_actual,  # Note opposite order
        )

        # Inverse-transform to coordinates suitable for folding:
        transform.inverse_transform_samples(samples)

        # Fold:
        folded = np.vectorize(
            transform.fold,
            signature=','.join('()' for _ in transform.sampled_params)+'->(n)'
        )(**samples[transform.sampled_params])

        # Rescale:
        with torch.no_grad():
            rescaled = self.parameter_rescaler.rescale(compressed_data, folded)
        return rescaled


def correct_right_ascension(samples, *, tgps_actual, tgps_fiducial):
    """
    Correct RA samples that had been obtained assuming a fiducial tgps.

    This will create a column 'lon' with the longitude, and change the
    value of 'ra' to be correct. By adding 'lon' we track that this
    function has been applied. If 'lon' is already present, a warning
    will be printed and the function will exit without changing
    anything.

    Parameters
    ----------
    samples : pandas.DataFrame
        Must contain a column 'ra' with the right ascension values to
        correct.

    tgps_actual : float
        GPS time at which the event actually happened.

    tgps_fiducial : float
        GPS time with which labrador was trained (``data_config.TGPS``).
    """
    if 'lon' in samples:
        warnings.warn('Refusing to change `samples["ra"]`, it looks like it '
                      'has already been corrected!')
        return

    gmst_actual = lal.GreenwichMeanSiderealTime(tgps_actual)
    gmst_fiducial = lal.GreenwichMeanSiderealTime(tgps_fiducial)
    samples['lon'] = cogwheel.skyloc_angles.ra_to_lon(samples['ra'],
                                                      gmst_fiducial)
    samples['ra'] = cogwheel.skyloc_angles.lon_to_ra(samples['lon'],
                                                     gmst_actual)


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


class ImportancePosterior:
    """Generate importance-weighted samples for a specific event."""
    def __init__(self, labrador_posterior, compressed_data, transform,
                 cogwheel_posterior):
        """
        Parameters
        ----------
        labrador_posterior : Posterior
            Amortized likelihood-free posterior estimator.

        compressed_data : array
            See :py:func:`labrador.compression.compress_data`.

        transform : transform.TransformMixin
            Maps folded_sampled_parameters to standard_parameters.

        cogwheel_posterior : cogwheel.posterior.Posterior
            Evaluates the (likelihood-based) posterior density in the
            space of standard parameters.
        """
        if (set(transform.standard_params)
                | labrador_posterior.fixed_par_dic.keys()
                != set(cogwheel_posterior.prior.standard_params)):
            raise ValueError('`transform` and `cogwheel_posterior` have '
                             'different `standard_params`.')

        self.labrador_posterior = labrador_posterior
        self.compressed_data = np.atleast_2d(compressed_data)
        self.transform = transform
        self.cogwheel_posterior = cogwheel_posterior

        self._standard_lnposterior_lnlike_v = np.vectorize(
            self._standard_lnposterior_lnlike, otypes=[float, float])

    def get_weighted_samples_and_lnz(
            self, target_n_eff=1000, max_n_samples=100_000):
        """
        Generate SBI samples, compute weights and log evidence.

        Parameters
        ----------
        target_n_eff, max_n_samples : int
            Keep drawing samples until an effective sample size of
            `target_n_eff` is reached or a total of `max_n_samples` has
            been drawn.

        Returns
        -------
        samples : pandas.DataFrame
            Weighted samples, where the weight is the ratio of the
            posterior (computed with cogwheel) to the normalizing-flow
            probability (computed with labrador).

        lnz : float
            Log evidence (Bayes factor vs. Gaussian noise).
        """
        n_chunk = target_n_eff
        n_eff = 0.0
        samples = pd.DataFrame()

        while (n_eff < target_n_eff) and (len(samples) < max_n_samples):
            n_chunk = min(n_chunk, max_n_samples - len(samples))
            chunk, standard_lnprob \
                = self.labrador_posterior.generate_samples_and_lnprob(
                    n_chunk, self.compressed_data, self.transform,
                    self.cogwheel_posterior.likelihood.event_data.tgps,
                    verbose=False)
            chunk['standard_lnprob'] = standard_lnprob
            chunk['standard_lnpost'], chunk['lnl'] \
                = self._standard_lnposterior_lnlike_v(
                    **chunk[self.cogwheel_posterior.prior.standard_params])
            chunk['ln_weights'] = chunk['standard_lnpost'] - standard_lnprob

            samples = pd.concat([samples, chunk.dropna()], ignore_index=True)

            n_eff = cogwheel.utils.n_effective(
                np.exp(samples['ln_weights'] - samples['ln_weights'].max()))

        # Z = ∫ π L = ⟨π L / p⟩_p
        lnz = logsumexp(samples['ln_weights']) - np.log(len(samples))
        samples['weights'] = np.exp(samples['ln_weights'] - lnz)

        print('Importance sampling efficiency: n_eff / N = '
              f'{n_eff:.1f} / {len(samples)} = {n_eff / len(samples):.3g}')
        return samples, lnz

    def _standard_lnposterior_lnlike(self, **standard_parameters):
        lnp = self.cogwheel_posterior.prior.standard_lnprior(
            **standard_parameters)
        lnl = self.cogwheel_posterior.likelihood.lnlike(standard_parameters)
        return lnp + lnl, lnl
