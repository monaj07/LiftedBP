from __future__ import print_function
from graph import Graph
import numpy as np
from sympy import *
import itertools
from collections import defaultdict
import copy


#############################################################################
def create_feature(join):
    """
    :param join: a superfeature which needs to be checked for the consistency of the arguments of its predicates
    :return: the projected and matched superfeature (if feature is not acceptable, returns [])
    """
    (vdom1, vdom2) = overlap_nodes(join[0].vdom, join[1].vdom)
    if vdom1 is None:
        return []
    #join[0].vdom = vdom1
    #join[1].vdom = vdom2
    ### There is no need to use these returned value, because join[0].vdom & join[1].vdom are internally changed within overlap_nodes function.
    # Just note that join[1].vdom that is provided as input to the next line, is already modified in previous function call and is not the same as
    # the initial join that is given to creat_feature function. But it is allright, since we want the overlap of all three join[0], join[1] and join[2].
    (vdom1, vdom2) = overlap_nodes(join[1].vdom, join[2].vdom)
    if vdom1 is None:
        return []
    #join[1].vdom = vdom1
    #join[2].vdom = vdom2
    #### NOTE: We might need to check the true_evidences and false evidences here as well
    return join
#----------------------------------------------------------------------------


#############################################################################
# This function is written specifically for rule = Smokes(x) ^ Friends(x,y) => Smokes(y)
def overlap_nodes(vdom1, vdom2):
    if vdom1==vdom2:
        return (vdom1, vdom2)
    dom1 = vdom1.domain
    dom2 = vdom2.domain
    dom1_new = {item1 for item1 in dom1 for item2 in dom2 if item1[-1]==item2[0]}
    dom2_new = {item2 for item1 in dom1 for item2 in dom2 if item1[-1]==item2[0]}
    if not dom1_new:
        return (None, None)
    vdom1.domain = dom1_new
    vdom2.domain = dom2_new
    return (vdom1, vdom2)
#----------------------------------------------------------------------------


#############################################################################
def returnVars(n):
    """
    :param n: the -ary of the predicate
    :return: for a n-ary predicate, returns the variable symbols
    """
    if n==1:
        return [Symbol('X')]
    elif n==2:
        return [Symbol('X'), Symbol('Y')]
    elif n==3:
        return [Symbol('X'), Symbol('Y'), Symbol('Z')]
    else:
        raise ValueError("Are you sure you want this many variables in your predicate?")
#----------------------------------------------------------------------------


#############################################################################
def return_different_worlds(narg, names):
    input = (tuple(names) for i in np.arange(narg))
    return tuple(itertools.product(*tuple(input)))
#----------------------------------------------------------------------------


#############################################################################
class varDom(object): # Defining the class that embeds variables and their constraints
    def __init__(self, v, domain = None):
        if domain is None:
            domain = set({})
        self.var = v
        self.domain = domain
    def __str__(self):
        st = ""
        for s in self.domain:
            st = st + "(" + ",".join(list(s)) + ") "
        return "%s" %(st)
    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               (self.domain==other.domain) # and self.var==other.var
    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
    def __hash__(self):
        return hash(tuple(self.domain))

class superNode(object):
    def __init__(self, pred, vdom):
        self.pred = pred
        self.vdom= vdom
    def __str__(self):
        return "(%s, %s)" %(self.pred, self.vdom)
    def print(self):
        pass
    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
               (self.pred==other.pred and self.vdom==other.vdom)
    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not(self == other)
    def __hash__(self):
        return hash((self.pred, self.vdom))

class superNodeEdgeWeight:
    def __init__(self, superNode, factorWeights):
        self.superNode = superNode
        self.factorWeight = factorWeights


#----------------------------------------------------------------------------


#############################################################################
def sampleLNC(evTable):
    """
    :param evTable: evTable is a dictionary where the keys are the predicates,
    and the value of each key is a list of tuples that provide evidences for different groundings.
    :return: The lifted graph
    """
    X = Symbol('X')
    Y = Symbol('Y')
    names = {"A", "B", "C", "D", "E", "F"}

    predicates = ["SMOKES", "FRIENDS"]
    rules = [["SMOKES", X, "FRIENDS", X, Y, "SMOKES", Y],]
    rules_parvars = (((X, ), (X, Y), (Y, )),)


    num_args = {"SMOKES":1, "FRIENDS":2}


    #############################################################################
    # FORMING THE INITIAL SUPERNODES, BASED ON THE EVIDENCE

    superNodes = dict()
    for pred in predicates:
        evidence_domain = set([])
        true_evdcs = []
        fals_evdcs = []
        if pred in evTable:
            v1 = varDom(Symbol('X'))# Initial supernode for True atoms
            v2 = varDom(Symbol('X'))# Initial supernode for false atoms
            evTuples = evTable[pred]
            for tp in evTuples:
                if tp[-1]==1: # True evidences. Assuming we are dealing with logical (binary) data, they have only two states, 0 & 1.
                    v1.domain.add(tp[0:-1])
                else:
                    v2.domain.add(tp[0:-1])
            if v1.domain:
                true_evdcs.append(superNode(pred, v1))
            if v2.domain:
                fals_evdcs.append(superNode(pred, v2))
            evidence_domain = v1.domain | v2.domain

        v = varDom(Symbol('X'))
        if num_args[pred]==1:
            v.domain = set(list(itertools.product(names)))
        elif num_args[pred]==2:
            v.domain = set(list(itertools.product(names,names)))
        else:
            ValueError("You have not written this code for predicates with more than two arguments!")

        v.domain = v.domain - evidence_domain
        unknowns = [superNode(pred, v)]
        superNodes[pred] = set(true_evdcs) | set(fals_evdcs) | set(unknowns)
    #----------------------------------------------------------------------------

    #############################################################################
    # Going through the main loop that successively builds and refines SuperNodes and SuperFeatures
    while 1:
        # Now, Forming the superfeatures by joining the supernodes
        superFeatures = []
        clause_vars = []
        for idx, rule in enumerate(rules):
            supfts = []
            supnds = []
            for pred in rule:
                if pred in predicates: # Discarding the noon-predicate items in rule, like X, Y, ...
                    supnds.append(superNodes[pred])
#                else: # in this case, pred contains the par-var symbols
            joins = tuple(itertools.product(*tuple(supnds)))
            for join in joins:
                joined_feat = create_feature(copy.deepcopy(join))
                if not joined_feat: # If the supernodes do not have consistent domains, they can not be "joined" and an empty list ([]) is returned.
                    continue
                supfts.append(joined_feat)
                clause_vars.append(idx)
            superFeatures.extend(supfts)

        # Now, Forming the supernodes by doing projection from superfeatures to predicates
        superNodesCounter = dict()
        for pred in predicates:
            if not pred in predicates: # Discarding the noon-predicate items in rule, like X, Y, ...
                continue
            parvar_groundings = return_different_worlds(num_args[pred], names)
            for parvar_grnd in parvar_groundings:
                projections = []
                for idx, supfeat in enumerate(superFeatures):
                    clause = rules_parvars[clause_vars[idx]] # has the form of ((X, ), (X, Y), (Y, )).
                    # Now, for each atom in the supfeat, we check if the atom.pred matches 'pred';
                    # if that is the case, we compute the number of projection of sufeat onto thia predicate with this grounding of 'paravar_grnd'.
                    proj = []
                    for i in np.arange(len(supfeat)):
                        if not supfeat[i].pred == pred:
                            proj.append(0)
                            continue
                        parvars = [item for tp in clause for item in tp] # parvars = [X, X, Y, Y]
                        parvarDict = dict.fromkeys(parvars, names) # Forming a dictionary that maps each par-var to its domain
                        pvs = clause[i]
                        for j in np.arange(num_args[pred]):
                            parvarDict[pvs[j]] = parvar_grnd[j] # Initializing the parvar_domain_dictionary with the groundings. Note that we want to compute all possible projection of this superfeature 'supfeat', onto this predicate with this specific grounding 'parvar_grnd'
                        for k in np.arange(len(supfeat)):
                            refined_dom = supfeat[k].vdom.domain
                            for ipv, pv in enumerate(clause[k]): # Loops through the par-vars within a predicate and makes their domains consistent with themselves and the grounding parvar_grnd
                                refined_dom = {item for item in refined_dom if item[ipv] in parvarDict[pv]}
                            for ipv, pv in enumerate(clause[k]):
                                parvarDict[pv] = {item[ipv] for item in refined_dom} # Restricting the domain of each par-var based on the refined domains of the atom

                        proj_num = 1
                        for k, v in parvarDict.items():
                            proj_num *= len(v)

                        proj.append(proj_num)
                    projections.append(tuple(proj))
                spnd_domain = set()
                spnd_domain.add(parvar_grnd)
                spnd = superNode(pred, varDom(None, spnd_domain))
                #print(spnd)
                superNodesCounter[spnd] = tuple(projections)
                #print("")

        #Now the superNodes (tuples) that have similar projection vectors are combined to form bigger superNodes with wider domains.
        superNodesUpdated = defaultdict(set)
        superNodesUpdatedWeights = defaultdict(set)
        projectionToTuplesDict = {}
        dummy = [projectionToTuplesDict.setdefault(v,[]).append(k) for (k,v) in superNodesCounter.items()]
        for k, v in projectionToTuplesDict.items():
            dom = set([])
            for velem in v:
                dom = dom | velem.vdom.domain
            sn = superNode(v[0].pred, varDom(None, dom))
            superNodesUpdated[v[0].pred].add(sn)
            superNodesUpdatedWeights[v[0].pred].add( (sn, tuple((item) for item in k) ) )

        if superNodesUpdated==superNodes:
            return superNodesUpdatedWeights
        # If superNodesUpdated==superNodes, then no more supernode has been added in this iteration of the main While loop, and we are done!
        # Else: The loop is not converged yet, go for another iteration for more refining of the superNodes.
        superNodes = superNodesUpdated
    # END OF WHILE
    #----------------------------------------------------------------------------

# END OF sampleLNC() FUNCTION
#----------------------------------------------------------------------------


def testToyGraph():

    evidences = {"SMOKES": [("A", 1)], "FRIENDS": [("B", "C", 1), ("C", "B", 1)]}
    superNodes = sampleLNC(evidences)
    [print(sn[0], " --- ", sn[1]) for spnodes in superNodes.values() for sn in spnodes]

# standard run of test cases
testToyGraph()
