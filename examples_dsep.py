import core


# +---+     +---+
# | X |     | Y |
# +---+     +---+
def bn_2_ind():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y'])
    return g

# +---+     +---+
# | X |---->| Y |
# +---+     +---+
def bn_2_dep():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y'])
    g.add_edge('X', 'Y')
    return g

# +---+     +---+     +---+
# | X |---->| Y |---->| Z |
# +---+     +---+     +---+
def bn_3_chain():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y', 'Z'])
    g.add_edges_from([('X', 'Y'), ('Y', 'Z')])
    return g

# +---+     +---+     +---+
# | Y |<----| X |---->| Z |
# +---+     +---+     +---+
def bn_3_naive():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y', 'Z'])
    g.add_edges_from([('X', 'Y'), ('X', 'Z')])
    return g

# +---+     +---+     +---+
# | X |---->| Z |<----| Y |
# +---+     +---+     +---+
def bn_3_vstruct():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y', 'Z'])
    g.add_edges_from([('X', 'Z'), ('Y', 'Z')])
    return g

# +---+          
# | X |          
# +---+          
#   |            
#   v            
# +---+     +---+
# | Y |<----| W |
# +---+     +---+
#   |         |  
#   v         |  
# +---+       |  
# | Z |<------+  
# +---+          
def bn_4_koller():
    g = core.BayesNet()
    g.add_nodes_from(['X', 'Y', 'W', 'Z'])
    g.add_edges_from([('X', 'Y'), ('W', 'Y'), ('W', 'Z'), ('Y', 'Z')])
    return g

# +------------+         +---------+
# | Earthquake |         | Burglar |
# +------------+         +---------+
#     |   |                   |
#     |   |     +-------+     |
#     |   +---->| Alarm |<----+
#     v         +-------+
# +-------+         |
# | Radio |         |
# +-------+         v
#               +-------+
#               | Phone |
#               +-------+
def bn_5_earthquake():
    g = core.BayesNet()
    g.add_nodes_from(['Earthquake', 'Burglar', 'Alarm', 'Radio', 'Phone'])
    g.add_edges_from([('Earthquake', 'Radio'),
                      ('Earthquake', 'Alarm'),
                      ('Burglar', 'Alarm'),
                      ('Alarm', 'Phone')])
    return g
