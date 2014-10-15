import math
import collections as col
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


EDGE_COLOR = '#bbbbbb'
EDGE_WIDTH = 2
NODE_SIZE = 3000
NODE_BORDER_COLOR = EDGE_COLOR
NODE_BORDER_WIDTH = 3
NODE_COLOR_NORMAL = '#3492d9'
NODE_COLOR_SOURCE = '#2cb64e'
NODE_COLOR_OBSERVED = '#d96b34'
NODE_COLOR_REACHABLE = NODE_COLOR_SOURCE
NODE_SHAPE_SOURCE = 'd'
LABEL_COLOR = '#111111'

class BayesNet(nx.DiGraph):
    def __init__(self):
        super(BayesNet, self).__init__()
        self.vs = {}
        self.cpts = {}

    def add_variable(self, name, domain):
        """Add a variable node to the network.

        Args:
          name(string): Variable name.
          domain(tuple): Values the variable can take.
        """
        name = str(name)
        v = Variable(name, domain)
        if name in self.vs:
            raise RuntimeError('Variable \'' + name + '\' already defined')
        self.vs[name] = v

    def add_cpt(self, parents, v, table):
        """Add a conditional probability table (CPT) to the network.

        Args:
          - parents(tuple): Parents of v in the network.
          - v(string): Variable for which the CPT is given.
          - table(dict): The CPT as a dictionary from tuples of variable values
            to conditional probabilities in the following form:

              {(vp_1, vp_2, ... , v_v): p, ...}
              
            In the above,  p is the conditional probability of v having value
            v_v, given that its parents have values vp_1, vp_2, etc.
        """
        self.cpts[v] = col.defaultdict(int, table)
        if parents == None:
            parents = []
        for p in parents:
            if not self.has_edge(p, v):
                self.add_edge(p, v)

    def get_ancestors(self, z):
        """Get all ancestors of all variables in z.

        Args:
          z (set): Set of variables.
        
        Returns:
          (set): The set of ancestors.
        """
        to_visit = set(z)
        anc = set()
        while to_visit != set():
            y = to_visit.pop()
            if y not in anc:
                anc.add(y)
                to_visit |= set(self.predecessors(y))
        return anc

    def get_reachable(self, x, z=None, plot=False):
        """Get all nodes that are reachable from x, given observed nodes z.
        
        Args:
          - x (string): Source node.
          - z (set, optional): A set of observed variables. Defaults to None,
            which corresponds to no observations.
          - plot(bool, optional): If True, plot network with distinguishing colors
            for observable, reachable, and d-separated nodes.

        Returns:
          (set): The set of reachable nodes.
        """
        if z == None: z = []
        z = set(z)
        assert x in set(self.nodes())
        assert z <= set(self.nodes())
        # First, find all ancestors of observed set z
        ancz = self.get_ancestors(z)
        # Then, perform breadth-first search starting from x
        to_visit = set([(x, False)])
        visited = set()
        reachable = set()
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
        # Plot
        if plot:
            self.draw(x, z, reachable)
        return reachable
        
    def draw(self, x=None, z=None, r=None):
        pos = nx.spectral_layout(self)
        nx.draw_networkx_edges(self, pos,
                               edge_color=EDGE_COLOR,
                               width=EDGE_WIDTH)
        rest = list(set(self.nodes()) - set([x]) - set(z) - set(r))
        if rest:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=rest,
                                         node_size=NODE_SIZE,
                                         node_color=NODE_COLOR_NORMAL)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        if x:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=[x],
                                         node_size=3000,
                                         node_color=NODE_COLOR_SOURCE,
                                         node_shape=NODE_SHAPE_SOURCE)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        if z:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=list(z),
                                         node_size=NODE_SIZE,
                                         node_color=NODE_COLOR_OBSERVED)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        if r:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=list(r),
                                         node_size=NODE_SIZE,
                                         node_color=NODE_COLOR_REACHABLE)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        nx.draw_networkx_labels(self, pos,
                                font_color=LABEL_COLOR)

class Variable:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
