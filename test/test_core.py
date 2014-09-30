import unittest2

from .. import core


def bn_2_ind():
    g = core.BayesNet()
    g.add_nodes_from([1, 2])
    return g

def bn_2_dep():
    g = core.BayesNet()
    g.add_nodes_from([1, 2])
    g.add_edge(1, 2)
    return g

def bn_3_chain():
    g = core.BayesNet()
    g.add_nodes_from([1, 2, 3])
    g.add_edges_from([(1, 2), (2, 3)])
    return g

def bn_3_naive():
    g = core.BayesNet()
    g.add_nodes_from([1, 2, 3])
    g.add_edges_from([(1, 2), (1, 3)])
    return g

def bn_3_vstruct():
    g = core.BayesNet()
    g.add_nodes_from([1, 2, 3])
    g.add_edges_from([(2, 1), (3, 1)])
    return g

def bn_4_koller():
    g = core.BayesNet()
    g.add_nodes_from([1, 2, 3, 4])
    g.add_edges_from([(1, 3), (2, 3), (2, 4), (3, 4)])
    return g

class TestDSeparation(unittest2.TestCase):
    def check_anc(self, g, z, correct):
        anc = g.get_ancestors(z)
        self.assertEqual(anc, set(correct))

    def test_anc_2_ind_1(self):
        self.check_anc(bn_2_ind(), [1], [1])

    def test_anc_2_ind_2(self):
        self.check_anc(bn_2_ind(), [2], [2])

    def test_anc_2_dep_1(self):
        self.check_anc(bn_2_dep(), [1], [1])

    def test_anc_2_dep_2(self):
        self.check_anc(bn_2_dep(), [2], [1, 2])

    def test_anc_3_chain_1(self):
        self.check_anc(bn_3_chain(), [1], [1])

    def test_anc_3_chain_2(self):
        self.check_anc(bn_3_chain(), [2], [1, 2])

    def test_anc_3_chain_3(self):
        self.check_anc(bn_3_chain(), [3], [1, 2, 3])

    def test_anc_3_naive_1(self):
        self.check_anc(bn_3_naive(), [1], [1])

    def test_anc_3_naive_2(self):
        self.check_anc(bn_3_naive(), [2], [1, 2])

    def test_anc_3_naive_3(self):
        self.check_anc(bn_3_naive(), [3], [1, 3])

    def test_anc_3_vstruct_1(self):
        self.check_anc(bn_3_vstruct(), [1], [1, 2, 3])

    def test_anc_3_vstruct_2(self):
        self.check_anc(bn_3_vstruct(), [2], [2])

    def test_anc_3_vstruct_3(self):
        self.check_anc(bn_3_vstruct(), [3], [3])

    def test_anc_4_koller_1(self):
        self.check_anc(bn_4_koller(), [1], [1])

    def test_anc_4_koller_2(self):
        self.check_anc(bn_4_koller(), [2], [2])

    def test_anc_4_koller_3(self):
        self.check_anc(bn_4_koller(), [3], [1, 2, 3])

    def test_anc_4_koller_4(self):
        self.check_anc(bn_4_koller(), [4], [1, 2, 3, 4])

    def test_anc_4_koller_1(self):
        self.check_anc(bn_4_koller(), [1], [1])

    def check_dsep(self, g, x, z, correct):
        dep = g.dsep(x, z)
        self.assertEqual(dep, set(correct))
