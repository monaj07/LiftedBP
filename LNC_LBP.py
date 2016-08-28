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
    names = {"A", "B", "C", "D"}#, "E", "F", "G", "H", "I", "J", "X", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W"}

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


#############################################################################
def makeTheLiftedGraph(superNodes):
    G = Graph()
    pot_SMK_FRNDS_SMK = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
    isn = 0 # Index of superNodes added to the graph.
    snodes = []
    snodes_projections = []
    snode_names = {}
    for predicate_spnodes in superNodes.values():
        for snode in predicate_spnodes:
            snd_name = "snd_" + str(isn)
            snd = G.addVarNode(snd_name, 2) # Binary entities
            snodes.append(snd)
            snode_names[snd_name] = snode[0].__str__()
            snodes_projections.append(snode[1]) #NOTE: 'snode' has two elements, first one is supernode that indicates the name of the predicate as well as the included groundings; secend element is the projections of the usperfeatures onto this supernode.
            isn += 1
    nsf = len(snodes_projections[0]) # number of superfeatures (taken from a sample supernode)
    nsn = len(snodes) # number of supernodes
    edge_weights_per_node = defaultdict(list)
    for i in np.arange(nsf):
        feat_nodes = [0]*len(snodes_projections[0][i]) # feat_nodes enlists the supernodes connected to the superfeature
        weights = [0]*len(snodes_projections[0][i])
        for j in np.arange(nsn):
            try:
                x = np.array(snodes_projections[j][i]) # snodes_projections[j][i] is a tuple that lists the projections of the superfeature i onto the supernode j
                index = np.where(x!=0)[0][0]
                feat_nodes[index] = j+1 # supernode j is connected to this superfeature as an atom at position 'index'. (added by 1 to avoid confusing supernode[j=0], with the case with no projection)
                #NOTE: Here I have specifically written the above for SMOKES(X)^FRIENDS(X,Y)=>SMOKES(Y). i.o. I have ignored the supernodes that exist in both sides of the superfeature
                # and only record their first occurrence. This later on results in a superfeature whose last element is zero and is discarded.
                # (Again, just for this clause. If you wanted to add more clauses, it might not be the same)
                weights[index] = x[index] # weight of each factor to node connection which comes from the projection numbers.
            except: # If there is no projection from the superfeature i onto supernode j, do nothing and check the next supernode
                continue
        if all(feat_nodes): # Ignoring the superfeatures where the factor connects to only two supernodes
            G.addFacNode(pot_SMK_FRNDS_SMK, snodes[feat_nodes[0]-1], snodes[feat_nodes[1]-1], snodes[feat_nodes[2]-1])
            for iw, w in enumerate(weights):
                edge_weights_per_node["snd_"+str(feat_nodes[iw]-1)].append(w) #For each supernodes, stores a weight. Note that the order of the weights that are listed for each supernode,
                # comes from the order of added factors to the graph. In other word, if you look at the 'var' element in graph G,
                # for each supernode var, a number neighboring factors 'nbrs' are listed which are ordered based on the order of their addition to the graph.
                # so we just use this fact to correspond the weights with the factors.
    G.addEdgeWeights(edge_weights_per_node) # The weights (the thickness of the connection between a supernode and its corresponding factor)
    return G, snode_names
# END OF makeTheLiftedGraph() FUNCTION
#----------------------------------------------------------------------------


def testLiftedGraph():

    evidences = {"SMOKES": [("A", 1)]}#, "FRIENDS": [("C", "D", 1), ("D", "C", 1)]}
    superNodes = sampleLNC(evidences)
    #[print("["+str(isn+1)+"]", sn[0], " --- ", sn[1]) for spnodes in superNodes.values() for isn, sn in enumerate(spnodes)]
    isn = 0
    for spnodes in superNodes.values():
        for sn in spnodes:
            print("["+str(isn)+"]", sn[0], " --- ", sn[1])
            isn += 1

    G, snode_names = makeTheLiftedGraph(superNodes)
#    G.var['snd_0'].condition(1)
    G.var['snd_1'].condition(1)

    print("------------------------------------")
    G.print_factors()

    marg = G.marginals()
    snd_marg = marg['snd_0']
    print("\nMarginals for " + snode_names["snd_0"] + " = ", snd_marg)
    snd_marg = marg['snd_2']
    print("\nMarginals for " + snode_names["snd_2"] + " = ", snd_marg)
    snd_marg = marg['snd_4']
    print("\nMarginals for " + snode_names["snd_4"] + " = ", snd_marg)

# standard run of test cases
testLiftedGraph()
