import numpy as np
import numpy.random as npr
import bprop


def cumavg(array, step=1):
    """Compute cumulative average of ``array``.

    The result is an array, such that, for all 0 <= i <= k*step

    cumavg[i] = (array[0] + array[step] + ... + array[k*step]) / (k + 1),

    where k = floor(len(array) / step).

    Arguments
    ---------
    array : iterable
        Input array.

    step : int
        Step size (only every step-th value of ``array`` is considered).

    Returns
    -------
    A (k + 1) x 1 array of cumulative averages.
    """
    array = np.array(array)
    avg = np.cumsum(array[::step, :], dtype=float)
    avg /= np.arange(1, len(avg) + 1)
    avg = np.reshape(avg, (-1, 1))
    return avg


def avg(array, step=1):
    """Compute the average of ``array`` taking into account only every
    ``step``-th value.
    """
    array = np.array(array)
    return np.reshape(np.mean(array[::step]), (-1, 1))


class GibbsSampler:
    def __init__(self, fgraph):
        self.fgraph = fgraph
        self.update_fgraph()

    def update_fgraph(self):
        """Should be called when the associated factor graph is updated."""
        self.vs = self.fgraph.vs
        self.vobs = self.fgraph.vobs

    def condition(self, observations):
        """Convenience method. Same as ``bprob.FactorGraph.condition``."""
        self.fgraph.condition(observations)
        self.update_fgraph()

    # NOTE: We could precompute/memoize the posteriors for more efficiency.
    def sample_var(self, v, state):
        """Sample a value of variable ``v`` from its posterior given ``state``.

        We need to only consider the values of variables in ``state`` that
        belong to the Markov blanket of ``v``, equivalently, all variables that
        participate in factors that are neighbors of ``v`` in the factor graph.

        Arguments
        ---------
        v : str
            Name of the variable to be updated.

        state : dict
            Current state of the Gibbs sampler.

        Returns
        -------
        A randomly sampled value of ``v`` from the posterior P(v | state\{v}).
        """
        v_domain = self.vs[v].domain
        prob = np.zeros(len(v_domain))
        for d in v_domain:
            for fnode in self.vs[v].neighbors:
                # Add variables one by one to create the combination that has
                # the same values as the current state for all other variables
                # in the factor and a value of d for variable v.
                comb = []
                for fnode_var in fnode.variables:
                    if fnode_var == v:
                        comb.append(d)
                    else:
                        comb.append(state[fnode_var])
                prob[d] += fnode.table[tuple(comb)]
        prob = bprop.normalize(prob)
        return npr.choice(v_domain, p=np.exp(prob))

    def run(self, niter, burnin=0, init_state=None, fcum=cumavg):
        """Run Gibbs sampler to obtain ``niter`` samples.

        Optionally, use a burn-in period during which samples are discarded,
        and specify (part of) the starting state. Furthermore, a function
        ``fcum`` for approximating marginals can be specified (see
        ``get_marginals`` for more details).

        Arguments
        ---------
        niter : int
            Number of samples to be returned.

        burnin : int
            Length of burn-in period.

        init_state : dict
            Starting state. Can be specified partially by only providing
            initial values for a subset of all variables.

        fcum : fun
            Function for computing marginals (see ``get_marginals``).

        Returns
        -------
        A tuple of computed marginals, variable domains, and observations,
        same as that returned by ``bprob.FactorGraph.run_bp``.
        """
        assert burnin < niter
        variables = self.vs.keys()
        samples = {v: [] for v in variables}
        # If not specified, the initial value of each variable is drawn
        # uniformly at random.
        state = {v: npr.choice(vnode.domain)
                 for v, vnode in self.vs.items()}
        if init_state is not None:
            state.update(init_state)
        # Burn-in period: samples are drawn and discarded.
        for it in range(burnin):
            current_variable = npr.choice(variables)
            state[current_variable] = self.sample_var(current_variable, state)
        # Actual recorded sampling.
        for it in range(niter):
            current_variable = npr.choice(variables)
            state[current_variable] = self.sample_var(current_variable, state)
            for v in variables:
                samples[v].append(state[v])
        marg = self.get_marginals(samples, fcum)
        domains = {v.name: v.orig_domain for v in self.vs.values()}
        return (marg, domains, self.fgraph.vobs)

    def get_marginals(self, samples, fcum):
        """Compute approximate marginals by applying ``fcum`` to ``samples``.

        For every value a variable can take, a binary indicator array is
        created that indicates at which iterations the variable had that value.
        Then, the given function ``fcum`` is applied to each of those binary
        arrays. ``fcum`` should return a r x 1 array (r >= 1).

        Arguments
        ---------
        samples : dict
            Dictionary that maps each variable to its samples as produced by
            the Gibbs sampler.

        fcum : fun
            A function that computes a r x 1 array per variable.

        Returns
        -------
        A dictionary that maps each variable v to a r x |domain(v)| array,
        i.e., each column contains the results of ``fcum`` for each value of v.
        """
        niter = len(samples.values()[0])
        assert niter >= 1
        marg = {v: None for v in samples}
        for v in samples:
            v_samples = np.array(samples[v])
            for d in self.vs[v].domain:
                bin_array = np.zeros((niter, 1))
                bin_array[v_samples == d] = 1
                cum = fcum(bin_array)
                if marg[v] is None:
                    marg[v] = cum
                else:
                    marg[v] = np.hstack((marg[v], cum))
        return marg
