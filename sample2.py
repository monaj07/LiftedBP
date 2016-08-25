from __future__ import print_function
from graphOld import Graph
import numpy as np

def makeToyGraph():
    '''
    Depressed(x) => Smokes(x)
    Smokes(x) => Cancer(x)
    '''
    G = Graph()
    d = G.addVarNode('d', 2)
    s = G.addVarNode('s', 2)
    c = G.addVarNode('c', 2)

    pot_ds = np.array([[4, 4], [1, 4]])
    pot_sc = np.array([[4, 4], [1, 4]])

    G.addFacNode(pot_ds, d, s)
    G.addFacNode(pot_sc, s, c)

    return G

def testToyGraph():

    G = makeToyGraph()
    G.var['s'].condition(1)
    marg = G.marginals()

    # check the marginals
    dM = marg['d']
    print("depressed marginals = ",dM)
    sM = marg['s']
    print("smoking marginals = ",sM)
    cM = marg['c']
    print("cancer marginals = ",cM)

# standard run of test cases
testToyGraph()
