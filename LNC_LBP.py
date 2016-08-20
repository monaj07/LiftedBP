from __future__ import print_function
from graph import Graph
import numpy as np
from sympy import *
import itertools


def sampleLNC(evTable):
    """
    :param evTable: evTable is a dictionary where the keys are the predicates,
    and the value of each key is a list of tuples that provide evidences for different groundings.
    :return: The lifted graph
    """
    X = Symbol('X')
    Y = Symbol('Y')
    names = {"A", "B", "C", "D"}

    predicates = ["SMOKES", "FRIENDS"]
    rules = [["SMOKES", X, "FRIENDS", X, Y, "SMOKES", Y]]

    #############################################################################
    # This function is written specifically for rule = Smokes(x) ^ Friends(x,y) => Smokes(y)
    def overlap_nodes(vcst1, vcst2):
        if vcst1==vcst2:
            return (vcst1, vcst2)
        dom1 = vcst1.domain
        dom2 = vcst2.domain
        dom1_new = {item1 for item1 in dom1 for item2 in dom2 if item1[-1]==item2[0]}
        dom2_new = {item2 for item1 in dom1 for item2 in dom2 if item1[-1]==item2[0]}
        if not dom1_new:
            return (None, None)
        vcst1.domain = dom1_new
        vcst2.domain = dom2_new
        return (vcst1, vcst2)
    #----------------------------------------------------------------------------


    #############################################################################
    def create_feature(join):
        """
        :param join: a superfeature which needs to be checked for the consistency of the arguments of its predicates
        :return: the projected and matched superfeature (if feature is not acceptable, returns [])
        """
        (vcst1, vcst2) = overlap_nodes(join[0].vcst, join[1].vcst)
        if vcst1 is None:
            return []
        join[0].vcst = vcst1
        join[1].vcst = vcst2

        (vcst1, vcst2) = overlap_nodes(join[1].vcst, join[2].vcst)
        if vcst1 is None:
            return []
        join[1].vcst = vcst1
        join[2].vcst = vcst2
        # NOTE: We might need to check the true_evidences and false evidences here as well
        return join
    #----------------------------------------------------------------------------


    num_args = {"SMOKES":1, "FRIENDS":2}

    pot_clause1 = np.array([[4, 4], [1, 4]])
    pot_clause2 = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
    pot_clause2_AT = np.array([[4, 4], [4, 4]])

    #############################################################################
    def returnVars(n):
        """
        :param n: the -ary of the predicate
        :return: for a n-ary predicate, returns the variable symbols
        """
        if n==1:
            return [Symbol('X')]
        elif n==2:
            return [Symbol('X'), Symbol('X')]
        elif n==3:
            return [Symbol('X'), Symbol('X'), Symbol('X')]
        else:
            raise ValueError("Are you sure you want this many variables in your predicate?")
    #----------------------------------------------------------------------------


    #############################################################################
    # FORMING THE INITIAL SUPERNODES, BASED ON THE EVIDENCE
    class varcsrt(object): # Defining the class that embeds variables and their constraints
        def __init__(self, v):
            self.var = v
            self.domain = None
        def __eq__(self, other):
            return isinstance(other, self.__class__) and \
                   (self.var==other.var & self.domain==other.domain)

    class superNode(object):
        def __init__(self, pred, vcst, status="U"):
            self.pred = pred
            self.vcst = vcst
            self.status = status
        def __eq__(self, other):
            return isinstance(other, self.__class__) and \
                   (self.pred==other.pred & self.vcst==other.vcst)# & self.status==other.status)

    superNodes = dict()
    for pred in predicates:
        true_evdcs = []
        fals_evdcs = []
        if pred in evTable:
            v1 = varcsrt(Symbol('X'))# Initial supernode for True atoms
            v2 = varcsrt(Symbol('X'))# Initial supernode for false atoms
            evTuples = evTable[pred]
            for tp in evTuples:
                if tp[-1]==1: # True evidences. Assuming we are dealing with logical (binary) data, they have only two states, 0 & 1.
                    v1.domain.update(tp[0:-1])
                else:
                    v2.domain.update(tp[0:-1])
            true_evdcs.append(superNode(pred, v1, "T"))
            fals_evdcs.append(superNode(pred, v2, "F"))

        v = varcsrt(Symbol('X'))
        if num_args[pred]==1:
            v.domain = itertools.product(names)
        elif num_args[pred]==2:
            v.domain = itertools.product(names,names)
        else:
            ValueError("You have not written this code for predicates with more than two arguments!")

        unknowns = [superNode(pred, v)]
        superNodes[pred] = true_evdcs+ fals_evdcs+ unknowns
    #----------------------------------------------------------------------------

    #############################################################################
    # Going through the main loop that successively builds and refines SuperNodes and SuperFeatures
    while 1:
        # Now, Forming the superfeatures by joining the supernodes
        superFeatures = []
        projections = [] # All atoms in all superfeatures. It is used when we want to project all superfeatures onto the predicates.
                    # In fact each element of this vector is a list that refers to a superfeature,
                    # for example if we have 50 superfeatures with 2 atoms and 10 suerfeatures with 3 stoms, the size of projections is 130.
        for rule in rules:
            supfts = []
            supnds = []
            for pred in rule:
                if pred in predicates:
                    supnds.append(superNodes[pred])
            joins = list(itertools.product(tuple(supnds)))
            for join in joins:
                joined_feat = create_feature(join)
                supfts += joined_feat
                projections.append(np.zeros(len(joined_feat)))
            superFeatures.extend(supfts)

        # Now, Forming the supernodes by doing projection from superfeatures to predicates
        superNodesCounter = dict()
        for pred in predicates:
            for supfeat in superFeatures:
                for i in np.arange(len(supfeat)):
                    tup = supfeat[i]
                    if pred == tup.pred:
                        # Here you need to compute the number of projections,
                        # which requires you to investigate the arguments of other predicates and count the number assignments for them,
                        # which in turn needs the variables ('X','Y',...) to be accompanied with their constraints.
                        if not tup in superNodesCounter:
                            superNodesCounter[tup] = projections

        for tup in superNodesCounter:
            for supfeat in superFeatures:
                for i in np.arange(len(supfeat)):
                    tup = supfeat[i]
                    for j in np.arange(len(supfeat)):
                        if j==i:
                            continue
                        tupNeighb = supfeat[j]
                        vcst = tupNeighb.vcst
                        pj = 1
                        for idx, grnd in enumerate(vcst.ground):
                            if not grnd:
                                pj = len(names) - len(vcst.domain[idx])
                        superNodesCounter[tup] = projections




    #----------------------------------------------------------------------------







def makeToyGraph():
    """
    Moves(x) => Walks(x)
    Walks(x) ^ Near(x,y) => Walks(y)
    """

    X = Symbol('X')
    Y = Symbol('Y')
    names = ["ana", "bob", "charles"]

    rules = [["WALKS", X, "NEAR", X, Y, "WALKS", Y]]
#    rules.append(["MOVES", X, "WALKS", X])

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
                        #print(var)
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
#    G.var['WALKS_ana'].condition(1)
    G.var['NEAR_bob_charles'].condition(1)
    G.var['NEAR_charles_bob'].condition(1)
    marg = G.marginals()

    # check the marginals
    mg = marg['WALKS_bob']
    print("WALKS(Bob) marginals = ",mg)
    mg = marg['WALKS_charles']
    print("WALKS(Charles) marginals = ",mg)
    mg = marg['NEAR_ana_charles']
    print("NEAR(ana, charles) marginals = ",mg)
    mg = marg['NEAR_ana_bob']
    print("NEAR(ana, bob) marginals = ",mg)
    print('\n')


    brute = G.bruteForce()
    wbbf = G.marginalizeBrute(brute, 'WALKS_bob')
    print("Walks(Bob) BRUTE_FORCE marginals = ",wbbf,"\n")

# standard run of test cases
testToyGraph()
