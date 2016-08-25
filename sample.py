from __future__ import print_function
from graphOld import Graph
import numpy as np

def makeToyGraph():
    '''
    Smokes(x) => Cancer(x)
    '''

    G = Graph()
    s = G.addVarNode('s', 2)
    c = G.addVarNode('c', 2)

    pot_sc = np.array([[4, 4], [1, 4]])
    G.addFacNode(pot_sc, s, c)
    return G

def testToyGraph():

    G = makeToyGraph()
    G.var['s'].condition(1)
    marg = G.marginals()
    brute = G.bruteForce()

    # check the marginals
    sM = marg['s']
    print("smoking marginals = ",sM)
    cM = marg['c']
    print("cancer marginals = ",cM)

    print("\nAll tests passed!")

# standard run of test cases
testToyGraph()
