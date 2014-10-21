import unittest2
from .. import core
from ..examples_dsep import *


class TestMisc(unittest2.TestCase):
    def test_check_cpt_valid_1(self):
        table = {(0, 0, 0): 0.999,
                 (0, 0, 1): 0.001,
                 (1, 0, 0): 0.00999,
                 (1, 0, 1): 0.99001,
                 (0, 1, 0): 0.98901,
                 (0, 1, 1): 0.01099,
                 (1, 1, 0): 0.0098901,
                 (1, 1, 1): 0.9901099}
        self.assertTrue(core.is_valid_cpt(table))

    def test_check_cpt_valid_2(self):
        table = {(0,): 1,
                 (1,): 0}
        self.assertTrue(core.is_valid_cpt(table))

    def test_check_cpt_invalid_1(self):
        table = {(0, 0, 0): 0.999,
                 (0, 0, 1): 0.001,
                 (1, 0, 0): 0.00999,
                 (1, 0, 1): 0.99001,
                 (0, 1, 0): 0.98901,
                 (0, 1, 1): 0.01099,
                 (1, 1, 0): 0.0098901,
                 (1, 1, 1): 0.99}
        self.assertFalse(core.is_valid_cpt(table))

    def test_check_cpt_invalid_2(self):
        table = {(0,): 0.5,
                 (1,): 0.4}
        self.assertFalse(core.is_valid_cpt(table))


class TestBayesNet(unittest2.TestCase):
    def test_double_variable(self):
        g = core.BayesNet()
        self.assertRaises(RuntimeError, g.add_variable('X', (0, 1)))


class TestDSeparation(unittest2.TestCase):
    def check_anc(self, g, z, correct):
        anc = g.get_ancestors(z)
        self.assertEqual(anc, set(correct))

    def test_anc_2_ind_1(self):
        self.check_anc(bn_independent(), ['X'], ['X'])

    def test_anc_2_ind_2(self):
        self.check_anc(bn_independent(), ['Y'], ['Y'])

    def test_anc_2_dep_1(self):
        self.check_anc(bn_dependent(), ['X'], ['X'])

    def test_anc_2_dep_2(self):
        self.check_anc(bn_dependent(), ['Y'], ['X', 'Y'])

    def test_anc_3_chain_1(self):
        self.check_anc(bn_chain(), ['X'], ['X'])

    def test_anc_3_chain_2(self):
        self.check_anc(bn_chain(), ['Y'], ['X', 'Y'])

    def test_anc_3_chain_3(self):
        self.check_anc(bn_chain(), ['Z'], ['X', 'Y', 'Z'])

    def test_anc_3_naive_1(self):
        self.check_anc(bn_naive_bayes(), ['X'], ['X'])

    def test_anc_3_naive_2(self):
        self.check_anc(bn_naive_bayes(), ['Y'], ['X', 'Y'])

    def test_anc_3_naive_3(self):
        self.check_anc(bn_naive_bayes(), ['Z'], ['X', 'Z'])

    def test_anc_3_vstruct_1(self):
        self.check_anc(bn_v_structure(), ['Z'], ['X', 'Y', 'Z'])

    def test_anc_3_vstruct_2(self):
        self.check_anc(bn_v_structure(), ['X'], ['X'])

    def test_anc_3_vstruct_3(self):
        self.check_anc(bn_v_structure(), ['Y'], ['Y'])

    def test_anc_4_koller_1(self):
        self.check_anc(bn_koller(), ['X'], ['X'])

    def test_anc_4_koller_2(self):
        self.check_anc(bn_koller(), ['W'], ['W'])

    def test_anc_4_koller_3(self):
        self.check_anc(bn_koller(), ['Y'], ['X', 'Y', 'W'])

    def test_anc_4_koller_4(self):
        self.check_anc(bn_koller(), ['Z'], ['X', 'W', 'Y', 'Z'])

    def check_reach(self, g, x, z, correct):
        dep = g.get_reachable(x, z)
        self.assertEqual(dep, set(correct))

    def test_reach_2_ind_1(self):
        self.check_reach(bn_independent(), 'X', None, [])

    def test_reach_2_ind_2(self):
        self.check_reach(bn_independent(), 'Y', None, [])

    def test_reach_3_chain_1(self):
        self.check_reach(bn_chain(), 'X', None, ['Y', 'Z'])

    def test_reach_3_chain_2(self):
        self.check_reach(bn_chain(), 'Z', None, ['X', 'Y'])

    def test_reach_3_chain_3(self):
        self.check_reach(bn_chain(), 'X', ['X'], [])

    def test_reach_3_chain_4(self):
        self.check_reach(bn_chain(), 'X', ['Y'], [])

    def test_reach_3_naive_1(self):
        self.check_reach(bn_naive_bayes(), 'X', None, ['Y', 'Z'])

    def test_reach_3_naive_2(self):
        self.check_reach(bn_naive_bayes(), 'Y', None, ['X', 'Z'])

    def test_reach_3_naive_3(self):
        self.check_reach(bn_naive_bayes(), 'Z', ['X'], [])

    def test_reach_3_vstruct_1(self):
        self.check_reach(bn_v_structure(), 'X', None, ['Z'])

    def test_reach_3_vstruct_2(self):
        self.check_reach(bn_v_structure(), 'X', ['Z'], ['Y'])

    def test_reach_4_koller_1(self):
        self.check_reach(bn_koller(), 'Z', ['Y'], ['X', 'W'])

    def test_reach_4_koller_2(self):
        self.check_reach(bn_koller(), 'Z', None, ['X', 'Y', 'W'])

    def test_reach_4_koller_3(self):
        self.check_reach(bn_koller(), 'W', ['Y'], ['X', 'Z'])
