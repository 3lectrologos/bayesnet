import unittest2
from .. import bprop
from ..examples_bprop import *


class TestBeliefPropagation(unittest2.TestCase):
    def test_earthquake_1(self):
        g = bprop.earthquake()
        g.condition({'Phone': 1})
        marg, _, _ = g.run_bp(10)
        self.assertAlmostEqual(marg['Burglar'][-1, 0], 0.505, places=3)

    def test_earthquake_2(self):
        g = bprop.earthquake()
        g.condition({'Phone': 1, 'Radio': 1})
        marg, _, _ = g.run_bp(10)
        self.assertAlmostEqual(marg['Burglar'][-1, 0], 0.917, places=3)
