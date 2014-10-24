import matplotlib.pyplot as plt
import bprop
import examples_bprop


# Load coin flipping Bayes network and plot it.
bn = examples_bprop.bn_naive_bayes()
bn.draw()
plt.show()

# Convert network to factor graph and plot it.
g = bprop.FactorGraph(bn)
g.draw()
plt.show()

# Compute marginal distributions for each variable over 10 iterations of belief
# propagations and plot them.
marg = g.run_bp(10)
bprop.draw_marginals(marg)
plt.show()

# Condition on having observed heads, tails, heads, compute the conditional
# marginal distributions, and plot them.
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
