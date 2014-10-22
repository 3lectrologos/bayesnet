import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import core
from conf import *


class FactorGraph:
    """An (undirected bipartite) factor graph with variable and factor nodes."""

    """Create a new factor graph or convert BayesNet ``bn`` to one, if given."""
    def __init__(self, bn=None):
        self.vs = {}
        self.fs = set()
        self.vobs = {}
        if bn != None:
            for v in bn.vs.values():
                self.add_variable(v.name, v.domain)
            for v in bn.vs.values():
                self.add_factor(list(v.parents) + [v.name], v.cpt)

    def add_variable(self, name, domain):
        """Add a variable node with the given name to the factor graph.

        Arguments
        ---------
        name : str
            Variable name.

        domain : iterable
            Values the variable can take.

        Returns
        -------
        The variable node that was added to the graph.
        """
        name = str(name)
        vnode = VariableNode(name, domain)
        if name in self.vs:
            raise RuntimeError("Variable '{0}' already defined".format(name))
        self.vs[name] = vnode
        return vnode

    def add_factor(self, variables, table):
        """Add a factor node to the factor graph.

        Arguments
        ---------
        variables : iterable of str
            Names of variables participating in the factor.

        table : dict
            The factor table as a dictionary from tuples of variable values to
            respective factor values in the following form:

              { (vp_1, vp_2, ... , vp_n): fv, ...}

            In the above, fv is the factor value when the participating
            variables have values vp_1, vp_2, ... , vp_n.

        Returns
        -------
        The factor node that was added to the graph.
        """
        unknown_vars = set(variables) - set(self.vs.keys())
        if unknown_vars != set():
            raise RuntimeError("Unknown variable '{0}'".format(
                unknown_vars.pop()))
        fnode = FactorNode(self, variables, table)
        self.fs.add(fnode)
        for v in variables:
            vnode = self.vs[v]
            vnode.connect_to(fnode)
            fnode.connect_to(vnode)
        return fnode

    def to_networkx(self):
        """Convert the factor graph to a networkx graph."""
        g = nx.Graph()
        for v in self.vs.values():
            g.add_node(v)
        for v in self.fs:
            g.add_node(v)
            for u in v.neighbors:
                g.add_edge(v, u)
        return g

    def draw(self):
        """Draw the factor graph."""
        g = self.to_networkx()
        pos = nx.spring_layout(g)
        nx.draw_networkx_edges(g, pos,
                               edge_color=EDGE_COLOR,
                               width=EDGE_WIDTH)
        obj = nx.draw_networkx_nodes(g, pos, nodelist=self.vs.values(),
                                     node_size=NODE_SIZE,
                                     node_color=NODE_COLOR_NORMAL)
        obj.set_linewidth(NODE_BORDER_WIDTH)
        obj.set_edgecolor(NODE_BORDER_COLOR)
        nx.draw_networkx_nodes(g, pos, nodelist=self.fs,
                               node_size=FACTOR_NODE_SIZE,
                               node_color=FACTOR_NODE_COLOR,
                               node_shape=FACTOR_NODE_SHAPE)
        nx.draw_networkx_labels(g, pos, {v: v.name
                                         for v in self.vs.values()},
                                font_color=LABEL_COLOR)

    def run_bp(self, niter):
        for v in self.vs.values():
            v.init_received()
        for f in self.fs:
            f.init_received()
        marg = {v: self.get_marginal(v) for v in self.vs}
        for it in range(niter):
            for v in self.vs.values():
                v.send()
            for f in self.fs:
                f.send()
            for v in self.vs:
                marg[v] = np.vstack((marg[v], self.get_marginal(v)))
        domains = {v.name: v.orig_domain for v in self.vs.values()}
        return (marg, domains, self.vobs)

    def condition(self, obs):
        unknown_vars = set(obs.keys()) - set(self.vs.keys())
        if unknown_vars != set():
            raise RuntimeError("Unknown variable '{0}'".format(
                 unknown_vars.pop()))
        self.vobs = obs
        for name, value in obs.iteritems():
            table = {(d,): 0 for d in self.vs[name].orig_domain}
            table[(value,)] = 1
            # Check if there is an existing factor that is only connected
            # to the observed variable. If that is the case, replace its
            # table by a new table corresponding to the observed value,
            # otherwise create a new factor with that table.
            found = False
            for fnode in self.fs:
                if len(fnode.variables) == 1 and fnode.variables[0] == name:
                    fnode.table = table
            if not found:
                fnode = self.add_factor((name,), table)

    def get_marginal(self, var):
        return self.vs[var].marginal()        


class Node(object):
    def __init__(self):
        self.neighbors = []
        self.received = {}

    def connect_to(self, node):
        self.neighbors.append(node)

    def send(self):
        for fnode in self.neighbors:
            self.send_one(fnode)

    def receive(self, source, msg):
        self.received[source] = msg


class VariableNode(Node):
    def __init__(self, name, domain):
        super(VariableNode, self).__init__()
        self.name = name
        # Map domain to nonnegative integers and store original domain, as well
        # as a domain map: original -> new.
        self.domain = range(len(domain))
        self.orig_domain = domain
        self.orig2new = dict(zip(domain, self.domain))

    def init_received(self):
        self.received = {fnode: np.zeros(len(self.domain))
                         for fnode in self.neighbors}

    def send_one(self, target):
        msg = np.zeros(len(self.domain))
        for fnode in self.neighbors:
            if fnode != target:
                msg += self.received[fnode]
        target.receive(self, normalize(msg))

    def marginal(self):
        m = np.zeros(len(self.domain))
        for fnode in self.neighbors:
            m += self.received[fnode]
        return np.exp(normalize(m))


class FactorNode(Node):
    def __init__(self, graph, variables, table):
        super(FactorNode, self).__init__()
        self.variables = variables
        # Map table combinations to numerical values.
        self.table = {}
        for comb, fvalue in table.iteritems():
            newcomb = tuple(graph.vs[v].orig2new[orig]
                            for v, orig in zip(variables, comb))
            self.table[newcomb] = fvalue
        self.name = 'F_' + reduce(lambda x, y: x + y,
                                  [v for v in variables],
                                  '')
        # Just to avoid annoying numpy warnings for log(0).
        for k, v in self.table.iteritems():
            if v == 0:
                self.table[k] = -1e6
            else:
                self.table[k] = np.log(v)

    def init_received(self):
        self.received = {}
                
    def send_one(self, target):
        # NOTE: Variable nodes in self.neighbors are in same order as in the
        # factor table tuples.
        target_index = self.neighbors.index(target)
        msg = -np.Inf*np.ones(len(target.domain))
        for comb, fvalue in self.table.iteritems():
            s = 0
            for i, vnode in enumerate(self.neighbors):
                if vnode != target:
                    s += self.received[vnode][comb[i]]
            s += fvalue
            msg[comb[target_index]] = np.logaddexp(msg[comb[target_index]], s)
        target.receive(self, msg)


def normalize(logdist):
    Z = reduce(np.logaddexp, logdist, -np.Inf)
    return logdist - Z


def draw_marginals(marg):
    marg, doms, obs = marg
    n = len(marg)
    t = len(marg.itervalues().next())
    rows = int(math.ceil(n/2.0))
    marg = sorted(marg.items())
    for i, (name, values) in enumerate(marg):
        if name in obs:
            plt.subplot(rows, 2, i+1, axisbg=AXIS_OBSERVED_BG_COLOR)
        else:
            plt.subplot(rows, 2, i+1)
        obj = plt.plot(values, '-o', linewidth=2, antialiased=True)
        for o in plt.gcf().findobj():
            o.set_clip_on(False)
        plt.ylim((0, 1))
        plt.legend(iter(obj), [name + '=' + str(d) for d in doms[name]])
