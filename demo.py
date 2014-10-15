import matplotlib.pyplot as plt
import core
import examples_dsep


g = examples_dsep.bn_5_earthquake()
g.get_reachable('Earthquake', plot=True)
plt.show()
g.get_reachable('Earthquake', ['Alarm'], plot=True)
plt.show()
