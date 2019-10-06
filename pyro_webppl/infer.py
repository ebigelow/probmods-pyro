import collections
import functools
import random
from six.moves import queue
from tqdm.auto import tqdm

import torch
from torch.autograd import Variable
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.abstract_infer import TracePosterior


def memoize(fn=None, **kwargs):
    if fn is None:
        return lambda _fn: memoize(_fn, **kwargs)
    return functools.lru_cache(**kwargs)(fn)


class EnumerateSearch(TracePosterior):
    """
    Exact inference by enumerating over all possible executions.  

    From RSA example.
    """
    def __init__(self, model, max_samples=1000, max_tries=int(1e6), method="BFS", **kwargs):
        self.model = model
        self.max_samples = max_samples
        self.max_tries = max_tries

        available_methods = ("BFS", "DFS", "best")
        method = method.upper()
        if method not in available_methods:
            raise ValueError("Method must be one of: {}".format(avail_methods))
        self.search_method = method

        super(EnumerateSearch, self).__init__(**kwargs)

    def _traces(self, *args, **kwargs):
        if self.search_method == "BFS":
            q = queue.LifoQueue()
        elif self.search_method == "DFS":
            q = queue.Queue()
        else:
            return self._traces_priority(*args, **kwargs)

        q.put(poutine.Trace())
        p = poutine.trace(
            poutine.queue(self.model, queue=q, max_tries=self.max_tries))
        
        n = 0
        pbar = tqdm(total=self.max_samples)

        while not q.empty() and n < self.max_samples:
            n += 1
            pbar.update(1)

            tr = p.get_trace(*args, **kwargs)
            yield tr, tr.log_prob_sum()

        pbar.close()


    def _traces_priority(self, *args, **kwargs):
        q = queue.PriorityQueue()

        # Add a little noise to priority queue to break ties
        q.put((torch.zeros(1).item() - torch.rand(1).item() * 1e-2, poutine.Trace()))
        q_fn = pqueue(self.model, queue=q)
        for i in tqdm(range(self.max_tries)):
            if q.empty():
                break
            tr = poutine.trace(q_fn).get_trace(*args, **kwargs)  # XXX should block
            yield tr, tr.log_prob_sum()



class RejectionSampling(TracePosterior):
    """
    Rejection sampling - generate examples until we find one that matches conditioned values.

    This assumes that we're conditioning by either:
        (1) pyro.sample('var', delta_like_function, obs=True)  
        (2) pyro.sample('var', delta_like_function(condition))
            ...
            pyro.condition(model, data={'var': True})
    """
    def __init__(self, model, num_samples=int(1e6), reject_log_prob=-1e10, **kwargs):
        self.model = model
        self.num_samples = num_samples
        self.reject_log_prob = reject_log_prob
        super(RejectionSampling, self).__init__(**kwargs)
        
    def _traces(self, *args, **kwargs):
        i = 0     # total number of samples
        t = 0     # non-rejected samples

        pbar = tqdm(total=self.num_samples)  # progress bar
        
        while t < self.num_samples:
            i += 1
            pbar.set_description("Total samples: {}".format(i))

            tr = poutine.trace(self.model).get_trace()
            tr.compute_log_prob()
            
            # If any observed nodes have log prob below threshold, reject this sample and continue
            reject = any(tr.nodes[n]["log_prob_sum"] < self.reject_log_prob for n in tr.observation_nodes)

            # Otherwise, increment and return the sample
            if not reject:
                t += 1
                pbar.update(1)
                yield tr, tr.log_prob_sum()

        pbar.close()


class MH(TracePosterior):
    """
    Simple implementation of Metropolis-Hastings algorithm.

    - Adopted from: https://github.com/pyro-ppl/pyro/blob/
        76097a8e0d9463c151a8590ec286fde99e5597ba/examples/storyboard/mh.py
    - https://forum.pyro.ai/t/general-implementation-of-metropolis-hastings/255/2
    """
    def __init__(self, model, guide=None, proposal=None, samples=10, lag=1, burn=0):
        super(MH, self).__init__()
        self.samples = samples
        self.lag = lag
        self.burn = burn
        self.model = model
        assert (guide is None or proposal is None) and \
            (guide is not None or proposal is not None), \
            "requires exactly one of guide or proposal, not both or none"
        if guide is not None:
            self.guide = lambda tr, *args, **kwargs: guide(*args, **kwargs)
        else:
            self.guide = proposal

    def _traces(self, *args, **kwargs):
        # Initialize traces with a draw from the prior
        old_model_trace = poutine.trace(self.model, *args, **kwargs).get_trace()

        traces = []
        t = 0
        i = 0

        total_iters = self.burn + (self.lag * self.samples)
        pbar = tqdm(total=total_iters)  # progress bar

        while t < total_iters:
            i += 1
            pbar.set_description("Total samples: {}".format(i))

            # q(z' | z)
            new_guide_trace = poutine.block(
                poutine.trace(self.guide).get_trace)(old_model_trace, *args, **kwargs)

            # p(x, z')
            new_model_trace = poutine.trace(
                poutine.replay(self.model, new_guide_trace)).get_trace(*args, **kwargs)

            # q(z | z')
            old_guide_trace = poutine.block(
                poutine.trace(
                    poutine.replay(self.guide, old_model_trace)).get_trace)(new_model_trace, 
                                                                            *args, **kwargs)

            # p(x, z') q(z' | z) / p(x, z) q(z | z')
            logr = new_model_trace.log_prob_sum() + new_guide_trace.log_prob_sum() - \
                   old_model_trace.log_prob_sum() - old_guide_trace.log_prob_sum()

            rnd = pyro.sample("mh_step_{}".format(i), dist.Uniform(0, 1))

            if torch.log(rnd) < logr:  # Accept
                t += 1
                pbar.update(1)

                old_model_trace = new_model_trace
                if t <= self.burn or (t > self.burn and t % self.lag == 0):
                    yield (new_model_trace, new_model_trace.log_prob_sum())

        pbar.close()


# Single-site
def single_site_proposal(model):
    def _fn(tr, *args, **kwargs):
        sample_sites = [s for s in tr.nodes.keys() 
                        if (tr.nodes[s]["type"] == "sample") and (s not in tr.observation_nodes)]

        choice_name = random.choice(sample_sites)
        return pyro.sample(choice_name, 
                           tr.nodes[choice_name]["fn"], 
                           *tr.nodes[choice_name]["args"], 
                           **tr.nodes[choice_name]["kwargs"])
    return _fn


class SingleSiteMH(MH):
    def __init__(self, model, **kwargs):
        super(SingleSiteMH, self).__init__(
            model, guide=None, proposal=single_site_proposal(model), **kwargs)


class HashingMarginal(dist.Distribution):
    """
    Inference in models that return a type that's not torch.Tensor; from RSA example.

    Note: currently not tested or well integrated with the rest of pmc_infer/pmc_webppl.

    :param trace_dist: a TracePosterior instance representing a Monte Carlo posterior
    """
    def __init__(self, trace_dist, sites=None):
        assert isinstance(trace_dist, TracePosterior), \
            "trace_dist must be trace posterior distribution object"

        if sites is None:
            sites = "_RETURN"

        assert isinstance(sites, (str, list)), \
            "sites must be either '_RETURN' or list"

        self.sites = sites
        super(HashingMarginal, self).__init__()
        self.trace_dist = trace_dist

    has_enumerate_support = True

    @memoize(maxsize=10)
    def _dist_and_values(self):
        # XXX currently this whole object is very inefficient
        values_map, logits = collections.OrderedDict(), collections.OrderedDict()
        for tr, logit in zip(self.trace_dist.exec_traces,
                             self.trace_dist.log_weights):

            # Get function return value
            if isinstance(self.sites, str):
                value = tr.nodes[self.sites]["value"]
            else:
                value = {site: tr.nodes[site]["value"] for site in self.sites}

            if not torch.is_tensor(logit):
                logit = torch.tensor(logit)

            # Hash value so we can use it as a dict key
            if torch.is_tensor(value):
                value_hash = hash(value.cpu().contiguous().numpy().tobytes())
            elif isinstance(value, dict):
                value_hash = hash(self._dict_to_tuple(value))
            else:
                value_hash = hash(value)

            # If value has already been seen  TODO
            if value_hash in logits:
                logits[value_hash] = dist.util.logsumexp(torch.stack([logits[value_hash], logit]), dim=-1)
            else:
                logits[value_hash] = logit
                values_map[value_hash] = value

        logits = torch.stack(list(logits.values())).contiguous().view(-1)
        logits = logits - dist.util.logsumexp(logits, dim=-1)
        d = dist.Categorical(logits=logits)
        return d, values_map

    def sample(self):
        d, values_map = self._dist_and_values()
        ix = d.sample()
        return list(values_map.values())[ix]

    def log_prob(self, val):
        d, values_map = self._dist_and_values()
        if torch.is_tensor(val):
            value_hash = hash(val.cpu().contiguous().numpy().tobytes())
        elif isinstance(val, dict):
            value_hash = hash(self._dict_to_tuple(val))
        else:
            value_hash = hash(val)
        return d.log_prob(torch.tensor([list(values_map.keys()).index(value_hash)]))

    def enumerate_support(self):
        d, values_map = self._dist_and_values()
        return list(values_map.values())

    def _dict_to_tuple(self, d):
        """
        Recursively converts a dictionary to a list of key-value tuples
        Only intended for use as a helper function inside HashingMarginal!!
        May break when keys cant be sorted, but that is not an expected use-case
        """
        if isinstance(d, dict):
            return tuple([(k, self._dict_to_tuple(d[k])) for k in sorted(d.keys())])
        else:
            return d

    def _weighted_mean(self, value, dim=0):
        weights = self._log_weights.reshape([-1] + (value.dim() - 1) * [1])
        max_weight = weights.max(dim=dim)[0]
        relative_probs = (weights - max_weight).exp()
        return (value * relative_probs).sum(dim=dim) / relative_probs.sum(dim=dim)

    @property
    def mean(self):
        samples = torch.stack(list(self._dist_and_values()[1].values()))
        return self._weighted_mean(samples)

    @property
    def variance(self):
        samples = torch.stack(list(self._dist_and_values()[1].values()))
        deviation_squared = torch.pow(samples - self.mean, 2)
        return self._weighted_mean(deviation_squared)


def EHMarginal(fn):
    """Simple syntax for enumerated hashing marginal; from RSA example."""
    return memoize(lambda *args: HashingMarginal(EnumerateSearch(fn).run(*args)))



class SequentialMonteCarlo(TracePosterior):
    """
    TODO: unfinished, untested

    - https://github.com/pyro-ppl/pyro/blob/dev/pyro/infer/smcfilter.py
    - https://github.com/pyro-ppl/pyro/blob/dev/examples/smcfilter.py
    """

    # IDEA: have wrapper model.step/guide.step and init functions, and require
    #   that we get model/guides with an observations arg.
    # With this wrapper, we'd then iterate through observations and have 
    #   pyro.sample("smc_step_{}".format(i), ...) for each
    pass


# class SimpleVariational(TracePosterior):
#     """
#     TODO

#     """
#     # ------------------------------------------------------------------------------------------------
#     # https://github.com/pyro-ppl/pyro/blob/
#     #    ce8f42b12ecd522e3ef7251e0d5f5175075a3fb4/examples/capture_recapture/cjs.py 
#     #
#     # we use poutine.block to only expose the continuous latent variables
#     # in the models to AutoDiagonalNormal (all of which begin with 'phi'
#     # or 'rho')
#     def expose_fn(msg):
#         return msg["name"][0:3] in ['phi', 'rho']

#     # we use a mean field diagonal normal variational distributions (i.e. guide)
#     # for the continuous latent variables.
#     guide = AutoDiagonalNormal(poutine.block(model, expose_fn=expose_fn))
#     # ------------------------------------------------------------------------------------------------


#     pass


