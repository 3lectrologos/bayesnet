import networkx as nx


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

    def dsep(self, x, z=None):
        if z == None: z = []
        z = set(z)
        assert x in set(self.nodes())
        assert z <= set(self.nodes())
