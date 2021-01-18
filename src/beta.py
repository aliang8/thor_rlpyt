
import torch

from numbers import Number
from rlpyt.utils.tensor import valid_mean
from rlpyt.distributions.base import Distribution
from torch.distributions.beta import Beta as TBeta
from rlpyt.utils.collections import namedarraytuple


DistInfo = namedarraytuple("DistInfo", ["mean"])
DistInfoBeta = namedarraytuple("DistInfoBeta", ["alpha", "beta"])

class Beta(Distribution):
    """Beta distribution class"""

    def __init__(
        self,
        dim,
        ):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def dirichlet(self, dist_info):
        alpha, beta = dist_info.alpha, dist_info.beta

        if isinstance(alpha, Number) and isinstance(beta, Number):
            params = torch.tensor([float(beta), float(alpha)])
        else:
            params = torch.stack([beta, alpha], -1)
        dirichlet = torch.distributions.Dirichlet(params)
        return dirichlet

    def sample(self, dist_info):
        """Generate random sample(s) from distribution informations."""
        alpha, beta = dist_info.alpha, dist_info.beta
        return TBeta(alpha, beta).sample()

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two distributions at each datum; should
        maintain leading dimensions (e.g. [T,B]).
        """
        raise NotImplementedError

    def mean_kl(self, old_dist_info, new_dist_info, valid):
        """
        Compute the mean KL divergence over a data batch, possible ignoring data
        marked as invalid.
        """
        raise valid_mean(self.kl(old_dist_info, new_dist_info), valid)

    def log_likelihood(self, x, dist_info):
        """
        Compute log-likelihood of samples ``x`` at distributions described in
        ``dist_info`` (i.e. can have same leading dimensions [T, B]).
        """
        alpha, beta = dist_info.alpha, dist_info.beta
        return TBeta(alpha, beta).log_prob(x)

        # dirichlet = self.dirichlet(dist_info)
        # if isinstance(x, Number):
        #     head_tails = torch.tensor([x, 1.0-x])
        # else:
        #     head_tails = torch.stack([x, 1.0-x])
        # return dirichlet.log_prob(head_tails)

        # log_x = torch.log(x)
        # log_one_minux_x = torch.log(1.-x)
        # log_gamma = torch.lgamma(alpha + beta) - torch.lgamma(alpha) - torch.lgamma(beta)
        # logli = log_x*(alpha-1.) + log_one_minux_x*(beta-1.)
        # logli = logli.sum(dim=1)
        # return logli

    def likelihood_ratio(self, x, old_dist_info, new_dist_info):
        """
        Compute likelihood ratio of samples ``x`` at new distributions over
        old distributions (usually ``new_dist_info`` is variable for
        differentiation); should maintain leading dimensions.
        """
        logli_old = self.log_likelihood(x, old_dist_info)
        logli_new = self.log_likelihood(x, new_dist_info)
        return torch.exp(logli_new - logli_old)

    def entropy(self, dist_info):
        """
        Compute entropy of distributions contained in ``dist_info``; should
        maintain any leading dimensions.
        """
        alpha, beta = dist_info.alpha, dist_info.beta
        return TBeta(alpha, beta).entropy()
        # dirichlet = self.dirichlet(dist_info)
        # return dirichlet.entropy()

    def perplexity(self, dist_info):
        """Exponential of the entropy, maybe useful for logging."""
        return torch.exp(self.entropy(dist_info))

    def mean_entropy(self, dist_info, valid=None):
        """In case some sophisticated mean is needed (e.g. internally
        ignoring select parts of action space), can override."""
        return valid_mean(self.entropy(dist_info), valid)

    def mean_perplexity(self, dist_info, valid=None):
        """Exponential of the entropy, maybe useful for logging."""
        return valid_mean(self.perplexity(dist_info), valid)


if __name__ == '__main__':
    dist = Beta(1)
    info = DistInfoBeta(-0.1, 0.1)
    print(dist.sample(info))
    print(dist.entropy(info))
    print(dist.log_likelihood(0.5, info))