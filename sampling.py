import math
import numpy as np
import numpy.random as npr
import matplotlib.pylab as plt
import bprop
import examples_bprop


class GibbsSampler:
    def __init__(self, fgraph):
        self.fgraph = fgraph
        self.vs = fgraph.vs

    # NOTE: These could be precomputed if we want to be more efficient.
    def sample_var(self, v, state):
        v_domain = self.vs[v].domain
        prob = np.zeros(len(v_domain))
        for d in v_domain:
            for fnode in self.vs[v].neighbors:
                # Create one by one the variable combination corresponding to
                # the current state for all other variables and a value of d
                # for the variable to be sampled.
                comb = []
                for fnode_var in fnode.variables:
                    if fnode_var == v:
                        comb.append(d)
                    else:
                        comb.append(state[fnode_var])
                prob[d] += fnode.table[tuple(comb)]
        prob = bprop.normalize(prob)
        return npr.choice(v_domain, p=np.exp(prob))

    def run(self, niter, burnin=0, init_state=None):
        assert burnin < niter
        variables = self.vs.keys()
        samples = {v: [] for v in variables}
        state = {v: npr.choice(vnode.domain)
                 for v, vnode in self.vs.items()}
        if init_state is not None:
            state.update(init_state)
        # Burn-in period: samples are discarded.
        for it in range(burnin):
            current_variable = npr.choice(variables)
            state[current_variable] = self.sample_var(current_variable, state)
        # Actual recorded sampling.
        for it in range(niter):
            current_variable = npr.choice(variables)
            state[current_variable] = self.sample_var(current_variable, state)
            for v in variables:
                samples[v].append(state[v])
        marg = self.get_marginals(samples)
        domains = {v.name: v.orig_domain for v in self.vs.values()}
        return (marg, domains, self.fgraph.vobs)

    def get_marginals(self, samples):
        niter = len(samples.values()[0])
        marg = {v: np.zeros((niter, 0)) for v in samples}
        # For each variable v, marg[v] is a niter x |domain(v)| matrix.
        for v in samples:
            samples_v = np.array(samples[v])
            for d in self.vs[v].domain:
                samples_d = np.zeros((niter, 1))
                samples_d[samples_v == d] = 1
                # Compute cumulative average, that is, for all 0 <= i < niter:
                #     cumavg[i] = (value_0 + ... + value_i) / (i + 1)
                cumavg = np.cumsum(samples_d, dtype=float)
                cumavg /= np.arange(1, len(cumavg) + 1)
                cumavg = np.reshape(cumavg, (-1, 1))
                marg[v] = np.hstack((marg[v], cumavg))
        return marg
