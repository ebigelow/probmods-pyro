
import matplotlib.pyplot as plt
import numpy as np

import torch
import pyro
import pyro.distributions as dist
import pyro.infer.mcmc
from pyro import poutine

import functools
import uuid
from pmc_infer import EnumerateSearch, RejectionSampling

def flip(p=0.5):
    return torch.distributions.Bernoulli(p).sample()

def pflip(name, p=0.5):
    return pyro.sample(name, dist.Bernoulli(p))

def repeat(f, n):
    return [f() for _ in range(n)]

def memoize(fn=None, **kwargs):
    if fn is None:
        return lambda _fn: memoize(_fn, **kwargs)
    return functools.lru_cache(**kwargs)(fn)

def factor(name, value):
    """
    Like factor in webPPL, adds a scalar weight to the log-probability of the trace
    """
    value = value if torch.is_tensor(value) else torch.Tensor([value])
    d = dist.Bernoulli(logits=value)
    pyro.sample(name, d, obs=torch.ones((value.size() or 1)))
    
def cond_var(name, value, normal_std=1e-15):
    return pyro.sample(name, dist.Normal(torch.tensor(value), torch.tensor(normal_std)))
    
def condition(name, var, value=1, normal_std=1e-15):
    pyro.sample(name, dist.Normal(torch.tensor(var), torch.tensor(normal_std)), 
                obs=torch.tensor(value))
    
def expectation(dist):
    return dist.mean



def Infer(model, 
          posterior_method='enumerate', 
          posterior_kwargs=dict(),
          num_samples=1000,
          draw_samples=False):
    """
    Pyro imitation of WebPPL's Infer operator.
    """
    if posterior_method == 'enumerate':
        posterior = EnumerateSearch(model, max_tries=num_samples, **posterior_kwargs)       # method: 'DFS' | 'BFS' 
    elif posterior_method in ('forward', 'importance'):
        posterior = pyro.infer.Importance(model, num_samples=num_samples, **posterior_kwargs)
    elif posterior_method.upper() == 'MCMC':
        kernel = pyro.infer.mcmc.HMC(model)
        posterior = pyro.infer.mcmc.MCMC(kernel, num_samples, **posterior_kwargs)  # warmup_steps, num_chains
    elif posterior_method == 'rejection':
        posterior = RejectionSampling(model, num_samples=num_samples, **posterior_kwargs)
    else:
        raise ValueError("Posterior method not defined for: {}".format(posterior_method))
        
    marginal = pyro.infer.EmpiricalMarginal(posterior.run())
    
    if draw_samples:
        samples = marginal.sample(torch.Size([num_samples]))
        return samples
    return marginal

def viz(data, to_type=(lambda v: v), plot_args={}, title=''):
    if type(data) in (list, torch.Tensor):
        # Histogram of samples
        data = [to_type(d) for d in data]
        plt.hist(data, weights=np.ones(len(data))/float(len(data)), **plot_args)

    elif isinstance(data, pyro.distributions.Distribution):
        # Barchart of distribution support
        d = {to_type(s): float(data.log_prob(s).exp()) for s in data.enumerate_support()}
        plt.bar(*zip(*d.items()), **plot_args)
    else:
        raise ValueError('data must be list of samples or pyro.distributions.Distribution')

    plt.title(title)
    plt.show()