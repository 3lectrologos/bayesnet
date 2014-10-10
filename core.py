import networkx as nx
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
        self.fs[fnode.name] = fnode
        for v in factor.variables:
            try:
                vnode = self.vs[v.name]
            except KeyError:
                vnode = VariableNode(v)
                self.vs[v.name] = vnode
            vnode.connect_to(fnode)
            fnode.connect_to(vnode)

    def to_networkx(self):
        g = nx.Graph()
        for v in self.vs.values():
            g.add_node(v.name, type='variable')
        for v in self.fs.values():
            g.add_node(v.name, type='factor')
            for u in v.neighbors:
                g.add_edge(v.name, u.name)
        return g

    def draw(self):
        g = self.to_networkx()
        pos = nx.graphviz_layout(g)
        nx.draw_networkx_edges(g, pos,
                               edge_color='#bbbbbb',
                               width=2)
        nx.draw_networkx_nodes(g, pos, nodelist=self.vs.keys(),
                               node_size=500,
                               node_color='#2c7bb6')
        nx.draw_networkx_labels(g, pos, {v: v for v in self.vs.keys()},
                                font_color='#eeeeee')
        nx.draw_networkx_nodes(g, pos, nodelist=self.fs.keys(),
                               node_size=200,
                               node_color='#fdae61',
                               node_shape='s')

class Variable:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain

class Factor:
    def __init__(self, variables, cpt):
        self.name = 'F_' + reduce(lambda x, y: x + y,
                                  [f.name for f in variables],
                                  '')
        self.variables = variables
        self.cpt = cpt

class Node(object):
    def __init__(self):
        self.neighbors = set()

    def connect_to(self, node):
        self.neighbors.add(node)

class VariableNode(Node):
    def __init__(self, variable):
        super(VariableNode, self).__init__()
        self.name = variable.name
        self.domain = variable.domain

class FactorNode(Node):
    def __init__(self, factor):
        super(FactorNode, self).__init__()
        self.name = factor.name
        self.variables = factor.variables
        self.cpt = factor.cpt

if __name__ == '__main__':
    g = FactorGraph()
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
    g.add_factor(f1)
    g.add_factor(f2)
    g.add_factor(f3)

    g.draw()
    plt.show()
