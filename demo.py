import matplotlib.pyplot as plt
import examples_dsep


g = examples_dsep.bn_earthquake()
g.get_reachable('Radio', plot=True)
plt.show()
g.get_reachable('Radio', ['Phone'], plot=True)
plt.show()
g.get_reachable('Radio', ['Phone', 'Earthquake'], plot=True)
plt.show()
