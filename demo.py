import matplotlib.pyplot as plt
import bprop
import examples_bprop


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
