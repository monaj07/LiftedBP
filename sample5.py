from __future__ import print_function
from graphOld import Graph
import numpy as np
from sympy import *

def makeToyGraph(names):
    """
    SMOKES(x) ^ FRIENDS(x,y) => SMOKES(y)
    """

    X = Symbol('X')
    Y = Symbol('Y')

    rules = [["SMOKES", X, "FRIENDS", X, Y, "SMOKES", Y]]

    num_args = {"SMOKES":1, "FRIENDS":2}

    pot_clause1 = np.array([[4, 4], [1, 4]])
    pot_clause2 = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
    pot_clause2_AT = np.array([[4, 4], [4, 4]])

    G = Graph()
    node_dict = {}
    fact_dict = {}

    for x in names:
        for y in names:
#            if x==y:
#                continue
            # Traverse the clause rule:
            for rule in rules:
                node_names = []
                ln = len(rule)
                c = 0
                nodes = []
                while c < ln:
                    pred = rule[c]
                    c += 1
                    narg = num_args[pred]
                    vars = []
                    for i in np.arange(narg):
                        var = rule[c].subs({X:x, Y:y})
#                        print(type(var))
#                        print(type(var.name))
                        c += 1
                        vars.append(var.name)
                    if (narg==1):
                        nd_name = pred+"_"+vars[0]
                        if not node_dict.get(nd_name, 0):
                            nd = G.addVarNode(nd_name,2)
                            node_dict[nd_name] = nd
                        else:
                            nd = node_dict[nd_name]
                    elif (narg==2):
                        nd_name = pred+"_"+vars[0]+"_"+vars[1]
                        if not node_dict.get(nd_name, 0):
                            nd = G.addVarNode(nd_name,2)
                            node_dict[nd_name] = nd
                        else:
                            nd = node_dict[nd_name]
                    else :
                        print("\nToo many arguments!!!\n")
                        exit(1)
                    nodes.append(nd)
                    node_names.append(nd.name)

                ground_clause = "_".join(node_names)
                if not (fact_dict.get(ground_clause,0)):
                    fact_dict[ground_clause] = 1
                else:
                    continue

                if len(nodes)==2 :
                    G.addFacNode(pot_clause1, nodes[0], nodes[1])
                elif len(nodes)==3 and not nodes[0]==nodes[2] :
                    G.addFacNode(pot_clause2, nodes[0], nodes[1], nodes[2])
                elif len(nodes)==3 and nodes[0]==nodes[2] :
                    G.addFacNode(pot_clause2_AT, nodes[0], nodes[1])


    return G

def testToyGraph():

    names = ["A", "B", "C", "D", "F"]
    G = makeToyGraph(names)
    G.var['SMOKES_A'].condition(1)
 #   G.var['SMOKES_B'].condition(1)
 #   G.var['SMOKES_C'].condition(1)
    G.var['FRIENDS_D_C'].condition(1)
    G.var['FRIENDS_C_D'].condition(1)
    marg = G.marginals()

    # check the marginals
    mg = marg['SMOKES_C']
    print("SMOKES(C) marginals = ",mg)
    mg = marg['SMOKES_B']
    print("SMOKES(B) marginals = ",mg)
    mg = marg['SMOKES_F']
    print("SMOKES(F) marginals = ",mg)
    mg = marg['FRIENDS_C_A']
    print("FRIENDS(C, A) marginals = ",mg)
    mg = marg['FRIENDS_A_B']
    print("FRIENDS(A, B) marginals = ",mg)
    mg = marg['FRIENDS_C_B']
    print("FRIENDS(C, B) marginals = ",mg)
    print('\n')


#    brute = G.bruteForce()
#    wbbf = G.marginalizeBrute(brute, 'SMOKES_bob')
#    print("SMOKES(Bob) BRUTE_FORCE marginals = ",wbbf,"\n")

    print("------------------------------------")

    G2 = Graph()

    b = G2.addVarNode('b', 2)
    c = G2.addVarNode('c', 2)
    fbc = G2.addVarNode('fbc', 2)
    fcb = G2.addVarNode('fcb', 2)

    pot_bc = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
    pot_cb = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
    P = 4*np.array([[1-0.31248069],[0.31248069]])

    G2.addFacNode(P, b)
    G2.addFacNode(P, c)
    G2.addFacNode(pot_bc, b, fbc, c)
    G2.addFacNode(pot_cb, c, fcb, b)

    G2.var['fbc'].condition(1)
    G2.var['fcb'].condition(1)

    marg = G2.marginals()
    mg = marg['b']
    print("smoking(b) = ",mg)
    mg = marg['c']
    print("smoking(c) = ",mg)
    mg = marg['fcb']
    print("friends(cb) = ",mg)

# standard run of test cases
testToyGraph()
