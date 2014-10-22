import matplotlib.pyplot as plt
import bprop
import examples_bprop


# Coin flipping network from Problem Set 2, Exercise 2.
bn = examples_bprop.bn_naive_bayes()
bn.draw()
plt.show()

g = bprop.FactorGraph(bn)
g.draw()
plt.show()

g.condition({'X1': 'H', 'X2': 'T', 'X3': 'H'})
marg = g.run_bp(10)
bprop.draw_marginals(marg)
plt.show()

# Earthquake network from Problem Set 3.
bn = examples_bprop.bn_earthquake()
bn.draw()
plt.show()

g = bprop.FactorGraph(bn)
g.draw()
plt.show()
marg = g.run_bp(10)
bprop.draw_marginals(marg)
plt.show()

g = bprop.FactorGraph(bn)
g.condition({'Phone': 1})
marg = g.run_bp(10)
bprop.draw_marginals(marg)
plt.show()

g = bprop.FactorGraph(bn)
g.condition({'Phone': 1, 'Radio': 1})
marg = g.run_bp(10)
bprop.draw_marginals(marg)
plt.show()
