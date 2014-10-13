import math
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


class BayesNet(nx.DiGraph):
    def __init__(self):
        super(BayesNet, self).__init__()

    def get_ancestors(self, z):
        to_visit = set(z)
        anc = set()
        while to_visit != set():
            y = to_visit.pop()
            if y not in anc:
                anc.add(y)
                to_visit |= set(self.predecessors(y))
        return anc

    def get_reachable(self, x, z=None):
        if z == None: z = []
        z = set(z)
        assert x in set(self.nodes())
        assert z <= set(self.nodes())
        to_visit = set([(x, False)])
        visited = set()
        reachable = set()
        ancz = self.get_ancestors(z)
        while to_visit != set():
            current = to_visit.pop()
            y, d = current
            if current in visited:
                continue
            if y not in z:
                reachable.add(y)
            visited.add(current)
            # Case of trail exiting y
            if d == False and y not in z:
                for p in self.predecessors_iter(y):
                    to_visit.add((p, False))
                for s in self.successors_iter(y):
                    to_visit.add((s, True))
            # Case of trail entering y
            elif d == True:
                if y not in z:
                    for s in self.successors_iter(y):
                        to_visit.add((s, True))
                elif y in ancz:
                    for p in self.predecessors_iter(y):
                        to_visit.add((p, False))
        # Just a convention to not return the query node
        reachable.discard(x)
        return reachable

class FactorGraph:
    def __init__(self):
        self.vs = dict()
        self.fs = dict()

    def add_factor(self, factor):
        fnode = FactorNode(factor)
        self.fs[factor] = fnode
        for v in factor.variables:
            try:
                vnode = self.vs[v]
            except KeyError:
                vnode = VariableNode(v)
                self.vs[v] = vnode
            vnode.connect_to(fnode)
            fnode.connect_to(vnode)

    def to_networkx(self):
        g = nx.Graph()
        for v in self.vs.values():
            g.add_node(v, type='variable')
        for v in self.fs.values():
            g.add_node(v, type='factor')
            for u in v.neighbors:
                g.add_edge(v, u)
        return g

    def draw(self):
        g = self.to_networkx()
        pos = nx.graphviz_layout(g)
        nx.draw_networkx_edges(g, pos,
                               edge_color='#bbbbbb',
                               width=2)
        nx.draw_networkx_nodes(g, pos, nodelist=self.vs.values(),
                               node_size=1000,
                               node_color='#2c7bb6')
        nx.draw_networkx_labels(g, pos, {v: v.variable.name
                                         for v in self.vs.values()},
                                font_color='#cccccc')
        nx.draw_networkx_nodes(g, pos, nodelist=self.fs.values(),
                               node_size=300,
                               node_color='#fdae61',
                               node_shape='s')

    def var(self, name):
        for v in self.vs:
            if v.name == name:
                return v
        raise KeyError('Variable not found in factor graph')

    def run_bp(self, niter):
        for v in self.vs.values():
            v.init_received()
        marg = {v: [self.get_marginal(v)] for v in self.vs}
        for it in range(niter):
            for v in self.vs.values():
                v.send()
            for v in self.fs.values():
                v.send()
            for v in self.vs:
                marg[v].append(self.get_marginal(v))
        return marg
        
    def condition(self, obs):
        for var, z in obs.iteritems():
            vals = {(d,): 0 for d in var.domain}
            vals[(z,)] = 1
            self.add_factor(Factor((var,), vals))

    def get_marginal(self, var):
        return self.vs[var].marginal()        

class Variable:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain

class Factor:
    def __init__(self, variables, cpt):
        self.name = 'F_' + reduce(lambda x, y: x + y,
                                  [v.name for v in variables],
                                  '')
        self.variables = variables
        self.cpt = cpt
        for k, v in self.cpt.iteritems():
            # Just to avoid annoying numpy warnings
            if v == 0:
                self.cpt[k] = -1e6
            else:
                self.cpt[k] = np.log(v)

class Node(object):
    def __init__(self):
        self.neighbors = set()
        self.received = {}

    def connect_to(self, node):
        self.neighbors.add(node)

    def send(self):
        for fnode in self.neighbors:
            self.send_one(fnode)

    def receive(self, source, msg):
        self.received[source] = msg

class VariableNode(Node):
    def __init__(self, variable):
        super(VariableNode, self).__init__()
        self.variable = variable

    def init_received(self):
        self.received = {fnode.factor: np.zeros(len(self.variable.domain))
                         for fnode in self.neighbors}

    def send_one(self, target):
        msg = np.zeros(len(self.variable.domain))
        for fnode in self.neighbors:
            if fnode != target:
                msg += self.received[fnode.factor]
        target.receive(self.variable, msg)

    def marginal(self):
        m = np.zeros(len(self.variable.domain))
        for fnode in self.neighbors:
            m += self.received[fnode.factor]
        Z = reduce(np.logaddexp, m, -np.Inf)
        return np.exp(m - Z)

class FactorNode(Node):
    def __init__(self, factor):
        super(FactorNode, self).__init__()
        self.factor = factor

    def send_one(self, target):
        target_index = self.factor.variables.index(target.variable)
        msg = -np.Inf*np.ones(len(target.variable.domain))
        for comb, fvalue in self.factor.cpt.iteritems():
            s = 0
            for i, vari in enumerate(self.factor.variables):
                if i != target_index:
                    s += self.received[vari][comb[i]]
            s += fvalue
            msg[comb[target_index]] = np.logaddexp(msg[comb[target_index]], s)
        target.receive(self.factor, msg)

def plot_marginals(marg):
    n = len(marg)
    t = len(marg.itervalues().next())
    rows = int(math.ceil(n/2.0))
    for i, var in enumerate(marg):
        plt.subplot(rows, 2, i+1)
        obj = plt.plot(marg[var], '-o')
        plt.ylim((0, 1))
        plt.legend(iter(obj), [var.name + '=' + str(d) for d in var.domain])
    plt.show()

def vstruct():
    x = Variable('X', (0, 1))
    y = Variable('Y', (0, 1))
    z = Variable('Z', (0, 1))
    f1 = Factor((x,), {(0,): 0.001, (1,): 0.999})
    f2 = Factor((y,), {(0,): 0.001, (1,): 0.999})
    f3 = Factor((x, y, z),
                {(0, 0, 0): 0.99,
                 (0, 0, 1): 0.01,
                 (0, 1, 0): 0.99,
                 (0, 1, 1): 0.01,
                 (1, 0, 0): 0.99,
                 (1, 0, 1): 0.01,
                 (1, 1, 0): 0.001,
                 (1, 1, 1): 0.999})
    g = FactorGraph()
    g.add_factor(f1)
    g.add_factor(f2)
    g.add_factor(f3)
    return g

def cycle():
    x = Variable('X', (0, 1))
    y = Variable('Y', (0, 1))
    z = Variable('Z', (0, 1))
    fxy = Factor((x, y),
                 {(0, 0): 13,
                  (0, 1): 17,
                  (1, 0): 14,
                  (1, 1): 25})
    fyz = Factor((y, z),
                 {(0, 0): 12,
                  (0, 1): 12,
                  (1, 0): 13,
                  (1, 1): 1})
    fzx = Factor((z, x),
                 {(0, 0): 15,
                  (0, 1): 14,
                  (1, 0): 13,
                  (1, 1): 20})
    g = FactorGraph()
    g.add_factor(fxy)
    g.add_factor(fyz)
    g.add_factor(fzx)
    return g

def two_nodes_three_values():
    x = Variable('X', (0, 1, 2))
    y = Variable('Y', (0, 1, 2))
    g = FactorGraph()
    g.add_factor(
        Factor((x, y),
               {(0, 0): 10,
                (0, 1): 5,
                (0, 2): 3,
                (1, 0): 2,
                (1, 1): 4,
                (1, 2): 1,
                (2, 0): 2,
                (2, 1): 12,
                (2, 2): 1
               }))
    return g

def earthquake():
    e = Variable('Earthquake', (0, 1))
    b = Variable('Burglar', (0, 1))
    r = Variable('Radio', (0, 1))
    a = Variable('Alarm', (0, 1))
    p = Variable('Phone', (0, 1))
    g = FactorGraph()
    g.add_factor(Factor((e,), {(0,): 0.999, (1,): 0.001}))
    g.add_factor(Factor((b,), {(0,): 0.999, (1,): 0.001}))
    g.add_factor(Factor((b, e, a),
                        {(0, 0, 0): 0.999,
                         (0, 0, 1): 0.001,
                         (1, 0, 0): 0.00999,
                         (1, 0, 1): 0.99001,
                         (0, 1, 0): 0.98901,
                         (0, 1, 1): 0.01099,
                         (1, 1, 0): 0.0098901,
                         (1, 1, 1): 0.9901099
                        }))
    g.add_factor(Factor((a, p),
                        {(0, 1): 0,
                         (0, 0): 1,
                         (1, 0): 0.5,
                         (1, 1): 0.5
                        }))
    g.add_factor(Factor((e, r),
                        {(0, 1): 0,
                         (0, 0): 1,
                         (1, 0): 0.2,
                         (1, 1): 0.8
                        }))
    return g

if __name__ == '__main__':
    g = earthquake()
    g.draw()
    plt.show()

    marg = g.run_bp(10)
    plot_marginals(marg)

    g.condition({g.var('Phone'): 1})
    marg = g.run_bp(10)
    plot_marginals(marg)

    g.condition({g.var('Radio'): 1})
    marg = g.run_bp(10)
    plot_marginals(marg)
