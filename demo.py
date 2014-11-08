import matplotlib.pyplot as plt
import bprop
import examples_bprop
import sampling


# Load coin flipping Bayes network and plot it.
bn = examples_bprop.bn_naive_bayes()
g = bprop.FactorGraph(bn)

# Create a Gibbs sampler and draw 20,000 samples, after a burn-in period
# of 1,000 samples.
sampler = sampling.GibbsSampler(g)
marg = sampler.run(20000, burnin=1000)

# Draw the cumulative average marginals after each sample.
bprop.draw_marginals(marg, markers=False)
plt.show()

# Do the same, when conditioned on having observed heads, tails, heads.
g.condition({'X1': 'H', 'X2': 'T', 'X3': 'H'})
sampler = sampling.GibbsSampler(g)
marg = sampler.run(20000, burnin=1000)
bprop.draw_marginals(marg, markers=False)
plt.show()

# Earthquake network from Problem Set 3.
bn = examples_bprop.bn_earthquake()
g = bprop.FactorGraph(bn)
sampler = sampling.GibbsSampler(g)
marg = sampler.run(20000, burnin=1000)
bprop.draw_marginals(marg, markers=False)
plt.show()

g = bprop.FactorGraph(bn)
g.condition({'Phone': 1})
sampler = sampling.GibbsSampler(g)
marg = sampler.run(20000, burnin=1000)
bprop.draw_marginals(marg, markers=False)
plt.show()

g = bprop.FactorGraph(bn)
g.condition({'Phone': 1, 'Radio': 1})
sampler = sampling.GibbsSampler(g)
marg = sampler.run(20000, burnin=1000)
bprop.draw_marginals(marg, markers=False)
plt.show()
