import numpy as np
import torch


class MixtureProposal:
    """Mixture of sbi posteriors that behaves like one."""
    def __init__(self, proposals, weights=None):
        if weights is None:
            weights = np.ones(len(proposals))
        elif len(weights) != len(proposals):
            raise ValueError(
                '`proposals` and `weights` must have the same length.')

        self.proposals = proposals
        self.weights = np.asarray(weights) / np.sum(weights)
        assert (self.weights >= 0).all()

        self._rng = np.random.default_rng()

    def sample(self, sample_shape, x, **kwargs):
        """
        sample_shape : torch.Size
            Desired shape of samples that are drawn from posterior.
        """
        n_tot = sample_shape[0]
        # Stratified sampling, top up with random:
        n_per_proposal = (n_tot * self.weights).astype(int)
        for ind in self._rng.choice(len(self.proposals),
                                    n_tot - n_per_proposal.sum()):
            n_per_proposal[ind] += 1

        samples_list = []
        for i, n in enumerate(n_per_proposal):
            samples_list.append(self.proposals[i].sample([n], x, **kwargs))

        return torch.cat(samples_list)

    def log_prob(self, theta, x):
        """
        Log-probability of the posterior, p(theta|x).

        Parameters
        ----------
        theta : torch.Tensor
            Parameters.

        x : torch.Tensor
            Data.
        """
        logp = torch.stack([p.log_prob(theta, x) for p in self.proposals])
        return torch.logsumexp(logp, dim=0)
