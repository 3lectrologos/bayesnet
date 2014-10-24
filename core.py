from collections import defaultdict
import networkx as nx
from conf import *


EPS = 1e-10


def is_valid_cpt(table):
    """Check that ``table`` contains valid conditional prob. distributions.

    Note that the conditional distributions are defined over the last variable
    in the tuples, while all other variables are conditioned on.
    """
    probabilities = defaultdict(float)
    for combination, value in table.items():
        if value < 0 or value > 1:
            return False
        probabilities[combination[:-1]] += value
    return all(abs(total - 1) <= EPS for total in probabilities.values())


class Variable:
    """A Bayesian network variable."""
    def __init__(self, name, domain, parents=None, cpt=None):
        self.name = name
        self.domain = domain
        self.parents = parents
        self.cpt = cpt


class BayesNet(nx.DiGraph):
    """A Bayesian network as a directed graph."""

    def __init__(self):
        super(BayesNet, self).__init__()
        self.vs = {}  # Variables of the network indexed by name.

    def add_variable(self, name, domain):
        """Add a variable node with the given name to the network.

        Arguments
        ---------
        name : str
            Variable name.

        domain : iterable
            Values the variable can take.
        """
        name = str(name)
        if name in self.vs:
            raise RuntimeError("Variable '{0}' already defined".format(name))
        v = Variable(name, domain, None, None)
        self.vs[name] = v

    def add_cpt(self, parents, variable, table):
        """Add a conditional probability table (CPT) to the network.

        Arguments
        ---------
        parents : iterable of str
            Parents of ``variable`` in the network.

        variable : str
            Variable for which the CPT is given.

        table : dict
            The CPT as a dictionary from tuples of variable values to
            conditional probabilities in the following form:

              { (vp_1, vp_2, ... , v_v): p, ...}

            In the above, p is the conditional probability of v having value
            v_v, given that its parents have values vp_1, vp_2, etc.
        """
        if parents is None:
            parents = ()
        elif isinstance(parents, str):
            parents = (parents,)
        else:
            parents = tuple(parents)
        for v in list(parents) + [variable]:
            if v not in self.vs:
                raise RuntimeError("Unknown variable '{0}'".format(v))
        # For CPTs with no parents, accept non-iterables as table keys for user
        # convenience, but convert them to single-element tuples here.
        newtable = {}
        for c, v in table.items():
            try:
                newtable[tuple(c)] = v
            except:
                newtable[(c,)] = v
        table = defaultdict(lambda: 0.5, newtable)
        if not is_valid_cpt(table):
            raise RuntimeError('Invalid CPT')
        self.vs[variable].parents = parents
        self.vs[variable].cpt = table
        for parent in parents:
            if not self.has_edge(parent, variable):
                self.add_edge(parent, variable)

    def draw(self, x=None, observed=None, dependent=None):
        """Draw the Bayesian network.

        Arguments
        ---------
        x : str
            The source variable.

        observed : iterable of str
            The variables on which we condition.

        dependent : iterable of str
            The variables which are dependent on ``x`` given ``observed``.
        """
        pos = nx.spectral_layout(self)
        nx.draw_networkx_edges(self, pos,
                               edge_color=EDGE_COLOR,
                               width=EDGE_WIDTH)
        if x or observed or dependent:
            rest = list(
                set(self.nodes()) - set([x]) - set(observed) - set(dependent))
        else:
            rest = self.nodes()
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
        if observed:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=list(observed),
                                         node_size=NODE_SIZE,
                                         node_color=NODE_COLOR_OBSERVED)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        if dependent:
            obj = nx.draw_networkx_nodes(self, pos, nodelist=list(dependent),
                                         node_size=NODE_SIZE,
                                         node_color=NODE_COLOR_REACHABLE)
            obj.set_linewidth(NODE_BORDER_WIDTH)
            obj.set_edgecolor(NODE_BORDER_COLOR)
        nx.draw_networkx_labels(self, pos, font_color=LABEL_COLOR)
