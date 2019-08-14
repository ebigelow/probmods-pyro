import random
from six.moves import queue
from tqdm import tqdm, tqdm_notebook

import torch
from torch.autograd import Variable
import pyro
from pyro import poutine
from pyro.distributions import Uniform
from pyro.infer.abstract_infer import TracePosterior


class EnumerateSearch(TracePosterior):
    """
    Exact inference by enumerating over all possible executions.
    """
    def __init__(self, model, max_tries=int(1e6), method='BFS', **kwargs):
        self.model = model
        self.max_tries = max_tries

        available_methods = ('BFS', 'DFS', 'best')
        if method not in available_methods:
            raise ValueError("Method must be one of: {}".format(avail_methods))
        self.search_method = method

        super(EnumerateSearch, self).__init__(**kwargs)

    def _traces(self, *args, **kwargs):
        if self.search_method == 'BFS':
            q = queue.LifoQueue()
        elif self.search_method == 'DFS':
            q = queue.Queue()
        else:
            return self._traces_priority(*args, **kwargs)

        q.put(poutine.Trace())
        p = poutine.trace(
            poutine.queue(self.model, queue=q, max_tries=self.max_tries))
        while not q.empty():
            tr = p.get_trace(*args, **kwargs)
            yield tr, tr.log_prob_sum()


    def _traces_priority(self, *args, **kwargs):
        q = queue.PriorityQueue()

        # Add a little noise to priority queue to break ties
        q.put((torch.zeros(1).item() - torch.rand(1).item() * 1e-2, poutine.Trace()))
        q_fn = pqueue(self.model, queue=q)
        for i in range(self.max_tries):
            if q.empty():
                break
            tr = poutine.trace(q_fn).get_trace(*args, **kwargs)  # TODO should block
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
        i = 0     # non-rejected samples
        n = 0     # total number of samples

        pbar = tqdm_notebook(total=self.num_samples)  # progress bar
        
        while i < self.num_samples:
            n += 1
            pbar.set_description('Total samples: {}'.format(n))

            tr = poutine.trace(self.model).get_trace()
            tr.compute_log_prob()
            
            # If any observed nodes have log prob below threshold, reject this sample and continue
            reject = any(tr.nodes[n]['log_prob_sum'] < self.reject_log_prob for n in tr.observation_nodes)

            # Otherwise, increment and return the sample
            if not reject:
                i += 1
                pbar.update(1)
                yield tr, tr.log_prob_sum()

        pbar.close()


class MH(TracePosterior):
    """
    Initial implementation of MH MCMC

    https://github.com/pyro-ppl/pyro/blob/76097a8e0d9463c151a8590ec286fde99e5597ba/examples/storyboard/mh.py
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
        # initialize traces with a draw from the prior
        old_model_trace = poutine.trace(self.model)(*args, **kwargs)
        traces = []
        t = 0
        i = 0
        while t < self.burn + self.lag * self.samples:
            i += 1
            # q(z' | z)
            new_guide_trace = poutine.block(
                poutine.trace(self.guide))(old_model_trace, *args, **kwargs)
            # p(x, z')
            new_model_trace = poutine.trace(
                poutine.replay(self.model, new_guide_trace))(*args, **kwargs)
            # q(z | z')
            old_guide_trace = poutine.block(
                poutine.trace(
                    poutine.replay(self.guide, old_model_trace)))(new_model_trace,
                                                                  *args, **kwargs)
            # p(x, z') q(z' | z) / p(x, z) q(z | z')
            logr = new_model_trace.log_pdf() + new_guide_trace.log_pdf() - \
                old_model_trace.log_pdf() - old_guide_trace.log_pdf()
            rnd = pyro.sample("mh_step_{}".format(i),
                              Uniform(pyro.zeros(1), pyro.ones(1)))

            if torch.log(rnd).data[0] < logr.data[0]:
                # accept
                t += 1
                old_model_trace = new_model_trace
                if t <= self.burn or (t > self.burn and t % self.lag == 0):
                    yield (new_model_trace, new_model_trace.log_pdf())

# Single-site
def single_site_proposal(model):
    def _fn(tr, *args, **kwargs):
        choice_name = random.choice(
            [s for s in tr.keys() if tr[s]["type"] == "sample"])
        return pyro.sample(choice_name,
                           tr[choice_name]["fn"],
                           *tr[choice_name]["args"][0],
                           **tr[choice_name]["args"][1])
    return _fn


class SingleSiteMH(MH):
    def __init__(self, model, **kwargs):
        super(SingleSiteMH, self).__init__(
            model, guide=None, proposal=single_site_proposal(model), **kwargs)
