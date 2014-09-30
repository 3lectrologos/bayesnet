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
