import unittest2
from .. import core
from ..examples_dsep import *


class TestBayesNet(unittest2.TestCase):
    def check_double_variable(self):
        g = core.BayesNet()
        g.add_variable('X', (0, 1))
        self.assertRaises(RuntimeError, g.add_variable('X', (0, 1)))

class TestDSeparation(unittest2.TestCase):
    def check_anc(self, g, z, correct):
        anc = g.get_ancestors(z)
        self.assertEqual(anc, set(correct))

    def test_anc_2_ind_1(self):
        self.check_anc(bn_2_ind(), ['X'], ['X'])

    def test_anc_2_ind_2(self):
        self.check_anc(bn_2_ind(), ['Y'], ['Y'])

    def test_anc_2_dep_1(self):
        self.check_anc(bn_2_dep(), ['X'], ['X'])

    def test_anc_2_dep_2(self):
        self.check_anc(bn_2_dep(), ['Y'], ['X', 'Y'])

    def test_anc_3_chain_1(self):
        self.check_anc(bn_3_chain(), ['X'], ['X'])

    def test_anc_3_chain_2(self):
        self.check_anc(bn_3_chain(), ['Y'], ['X', 'Y'])

    def test_anc_3_chain_3(self):
        self.check_anc(bn_3_chain(), ['Z'], ['X', 'Y', 'Z'])

    def test_anc_3_naive_1(self):
        self.check_anc(bn_3_naive(), ['X'], ['X'])

    def test_anc_3_naive_2(self):
        self.check_anc(bn_3_naive(), ['Y'], ['X', 'Y'])

    def test_anc_3_naive_3(self):
        self.check_anc(bn_3_naive(), ['Z'], ['X', 'Z'])

    def test_anc_3_vstruct_1(self):
        self.check_anc(bn_3_vstruct(), ['Z'], ['X', 'Y', 'Z'])

    def test_anc_3_vstruct_2(self):
        self.check_anc(bn_3_vstruct(), ['X'], ['X'])

    def test_anc_3_vstruct_3(self):
        self.check_anc(bn_3_vstruct(), ['Y'], ['Y'])

    def test_anc_4_koller_1(self):
        self.check_anc(bn_4_koller(), ['X'], ['X'])

    def test_anc_4_koller_2(self):
        self.check_anc(bn_4_koller(), ['W'], ['W'])

    def test_anc_4_koller_3(self):
        self.check_anc(bn_4_koller(), ['Y'], ['X', 'Y', 'W'])

    def test_anc_4_koller_4(self):
        self.check_anc(bn_4_koller(), ['Z'], ['X', 'W', 'Y', 'Z'])

    def check_reach(self, g, x, z, correct):
        dep = g.get_reachable(x, z)
        self.assertEqual(dep, set(correct))

    def test_reach_2_ind_1(self):
        self.check_reach(bn_2_ind(), 'X', None, [])

    def test_reach_2_ind_2(self):
        self.check_reach(bn_2_ind(), 'Y', None, [])

    def test_reach_3_chain_1(self):
        self.check_reach(bn_3_chain(), 'X', None, ['Y', 'Z'])

    def test_reach_3_chain_2(self):
        self.check_reach(bn_3_chain(), 'Z', None, ['X', 'Y'])

    def test_reach_3_chain_3(self):
        self.check_reach(bn_3_chain(), 'X', ['X'], [])

    def test_reach_3_chain_4(self):
        self.check_reach(bn_3_chain(), 'X', ['Y'], [])

    def test_reach_3_naive_1(self):
        self.check_reach(bn_3_naive(), 'X', None, ['Y', 'Z'])

    def test_reach_3_naive_2(self):
        self.check_reach(bn_3_naive(), 'Y', None, ['X', 'Z'])

    def test_reach_3_naive_3(self):
        self.check_reach(bn_3_naive(), 'Z', ['X'], [])

    def test_reach_3_vstruct_1(self):
        self.check_reach(bn_3_vstruct(), 'X', None, ['Z'])

    def test_reach_3_vstruct_2(self):
        self.check_reach(bn_3_vstruct(), 'X', ['Z'], ['Y'])

    def test_reach_4_koller_1(self):
        self.check_reach(bn_4_koller(), 'Z', ['Y'], ['X', 'W'])

    def test_reach_4_koller_2(self):
        self.check_reach(bn_4_koller(), 'Z', None, ['X', 'Y', 'W'])

    def test_reach_4_koller_3(self):
        self.check_reach(bn_4_koller(), 'W', ['Y'], ['X', 'Z'])
