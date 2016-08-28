import itertools
import numpy as np
import scipy
import graphOld as Graph

a = {1,2,3}
combs = list(itertools.product(a , a))
print(combs)


G2 = Graph()

b = G2.addVarNode('b', 2)
c = G2.addVarNode('c', 2)
fbc = G2.addVarNode('fbc', 2)
fcb = G2.addVarNode('fcb', 2)

pot_bc = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
pot_cb = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
#    P = 4*np.array([[1-0.38461538],[0.38461538]])

#    G2.addFacNode(P, b)
#    G2.addFacNode(P, c)
G2.addFacNode(pot_bc, b, fbc, c)
G2.addFacNode(pot_cb, c, fcb, b)

marg = G2.marginals()
mg = marg['b']
print("smoking(b) = ",mg)
mg = marg['c']
print("smoking(c) = ",mg)