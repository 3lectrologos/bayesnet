import unittest2
from ..examples_bprop import bn_earthquake
from ..bprop import FactorGraph


class TestBeliefPropagation(unittest2.TestCase):
    def test_earthquake_1(self):
        g = bn_earthquake()
        fg = FactorGraph(g)
        fg.condition({'Phone': 1})
        marg, _, _ = fg.run_bp(10)
        self.assertAlmostEqual(marg['Burglar'][-1, 0], 0.505, places=3)

    def test_earthquake_2(self):
        g = bn_earthquake()
        fg = FactorGraph(g)
        fg.condition({'Phone': 1, 'Radio': 1})
        marg, _, _ = fg.run_bp(10)
        self.assertAlmostEqual(marg['Burglar'][-1, 0], 0.917, places=3)
