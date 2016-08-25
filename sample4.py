
"""
Ground MLN belief propagation inference
"""

from __future__ import print_function
from graphOld import Graph
import numpy as np
from sympy import *

def makeToyGraph():
    """
    Moves(x) => Walks(x)
    Walks(x) ^ Near(x,y) => Walks(y)
    """

    X = Symbol('X')
    Y = Symbol('Y')
    names = ["ali", "bob", "sam", "joe", "man", "kkk", "ddd", "eee", "www", "qqq", "sss", "AAA", "FFF", "TTT", "ED", "SX", "VF", "WQ"]

    rules = [["WALKS", X, "NEAR", X, Y, "WALKS", Y]]
    rules.append(["MOVES", X, "WALKS", X])

    num_args = {"WALKS":1, "MOVES":1, "NEAR":2}

    pot_clause1 = np.array([[4, 4], [1, 4]])
    pot_clause2 = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
    pot_clause2_AT = np.array([[4, 4], [4, 4]])

    G = Graph()
    node_dict = {}
    fact_dict = {}

    for x in names:
        for y in names:
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

    G = makeToyGraph()
    G.var['WALKS_ali'].condition(1)
#    G.var['NEAR_bob_sam'].condition(1)
#    G.var['NEAR_sam_bob'].condition(1)
    marg = G.marginals()

    mg = marg['WALKS_bob']
    print("WALKS(bob) marginals = ",mg)
    mg = marg['WALKS_sam']
    print("WALKS(sam) marginals = ",mg)
    mg = marg['WALKS_joe']
    print("WALKS(joe) marginals = ",mg)

    # check the marginals
    mb_marg = marg['MOVES_bob']
    print("Moves(Bob) marginals = ",mb_marg)
    """
    nba_marg = marg['nba']
    print("Near(Bob,Ali) marginals = ",nba_marg)
    naa_marg = marg['naa']
    print("Near(Ali,Ali) marginals = ",naa_marg)
    ma_marg = marg['ma']
    print("Moves(Ali) marginals = ",ma_marg)
    mb_marg = marg['mb']
    print("Moves(Bob) marginals = ",mb_marg)
    """
    wa_marg = marg['WALKS_ali']
    print("Walks(Ali) marginals = ",wa_marg)
    wb_marg = marg['WALKS_bob']
    print("Walks(Bob) marginals = ",wb_marg)
    wb_marg = marg['WALKS_sam']
    print("Walks(Sam) marginals = ",wb_marg)
    print('\n')


#    brute = G.bruteForce()
#    wbbf = G.marginalizeBrute(brute, 'WALKS_bob')
#    print("Walks(Bob) BRUTE_FORCE marginals = ",wbbf,"\n")

# standard run of test cases
testToyGraph()
