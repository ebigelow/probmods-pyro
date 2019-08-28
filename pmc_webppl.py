
import matplotlib.pyplot as plt
import numpy as np

import torch
import pyro
import pyro.distributions as dist
import pyro.infer.mcmc
from pyro import poutine

from pmc_infer import memoize, EnumerateSearch, RejectionSampling, SingleSiteMH, HashingMarginal


def maybe_tensor(v):
    return v if torch.is_tensor(v) else torch.Tensor([v])


def flip(p=0.5):
    return torch.distributions.Bernoulli(p).sample()


def pflip(name, p=0.5):
    return pyro.sample(name, dist.Bernoulli(p))


def repeat(f, n):
    return [f() for _ in range(n)]


def factor(name, value):
    """
    Like factor in webPPL, adds a scalar weight to the log-probability of the trace

    TODO: this is now a pyro built-in  
          https://github.com/pyro-ppl/pyro/commit/86ea9051a6174acbb4c2f498bb486c41d464b5eb
    """
    value = maybe_tensor(value)
    d = dist.Bernoulli(logits=value)
    pyro.sample(name, d, obs=torch.ones((value.size() or 1)))
    
def cond_var(name, value, normal_std=1e-7):
    return pyro.sample(name, dist.Normal(torch.tensor(value, dtype=torch.float), torch.tensor(normal_std)))
    
def condition(name, var, value=1.0, normal_std=1e-7):
    var = maybe_tensor(var)
    normal_std = maybe_tensor(normal_std)
    value = maybe_tensor(value)
    pyro.sample(name, dist.Normal(var, normal_std), obs=value)
    
def expectation(dist):
    return dist.mean


def viz(data, to_type=(lambda v: v), plot_args={}, title=""):
    """data is either Tensor/list of samples, or dist for which we enumerate samples and probs."""
    if type(data) in (list, torch.Tensor):
        # Histogram of samples
        data = [to_type(d) for d in data]
        plt.hist(data, weights=np.ones(len(data))/float(len(data)), **plot_args)

    elif isinstance(data, pyro.distributions.Distribution):
        # Barchart of distribution support

        support = data.enumerate_support()
        if type(support) is list:
            support = torch.tensor(support, dtype=support[0].dtype)
        else:
            support = support.unique()

        d = {to_type(s): float(data.log_prob(s).exp()) for s in support}
        plt.bar(*zip(*d.items()), **plot_args)
    else:
        raise ValueError("data must be list of samples or pyro.distributions.Distribution")

    plt.title(title)
    plt.show()


def viz_table(args):
    # TODO
    return

def viz_scatter(data, args):
    # TODO
    return


def Infer(model, 
          posterior_method="enumerate", 
          posterior_kwargs={},
          marginal_method="empirical",
          marginal_kwargs={},
          num_samples=int(1e5),
          draw_samples=False):
    """
    Pyro imitation of WebPPL's Infer operator.

    WebPPL methods not yet implemented
    - HMC (partially done)
    - SMC
    - incremental MH
    - optimize (variational): sgd, adagrad, rmsprop, adam

    Ref: https://github.com/probmods/webppl/blob/dev/docs/inference/methods.rst
    """
    p_method = posterior_method.lower()
    if p_method == "enumerate":
        posterior = EnumerateSearch(model, max_samples=num_samples, **posterior_kwargs)       # method: "DFS" | "BFS" 
    elif p_method in ("forward", "importance"):
        posterior = pyro.infer.Importance(model, num_samples=num_samples, **posterior_kwargs)
    elif p_method == "rejection":
        posterior = RejectionSampling(model, num_samples=num_samples, **posterior_kwargs)
    elif p_method == "mh":
        posterior = SingleSiteMH(model, samples=num_samples, **posterior_kwargs)
    elif p_method == "hmc":
        # currently buggy
        kernel = pyro.infer.mcmc.HMC(model)
        posterior = pyro.infer.mcmc.MCMC(kernel, num_samples, **posterior_kwargs)  # warmup_steps, num_chains
    else:
        raise ValueError("Posterior method not defined for: {}".format(posterior_method))

    posterior = posterior.run()
        
    m_method = marginal_method.lower()
    if m_method == "empirical":
        marginal = pyro.infer.EmpiricalMarginal(posterior, **marginal_kwargs)
    elif m_method in ("hash", "hashing"):
        marginal = HashingMarginal(posterior, **marginal_kwargs)
    else:
        raise ValueError("Posterior method not defined for: {}".format(marginal_method))
    
    if draw_samples:
        samples = marginal.sample(torch.Size([num_samples]))
        return samples
    return marginal
