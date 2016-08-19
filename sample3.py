from __future__ import print_function
from graph import Graph
import numpy as np

def makeToyGraph():
    """
    Moves(x) => Walks(x)
    Walks(x) ^ Near(x,y) => Walks(y)
    """
    G = Graph()
    m_a = G.addVarNode('ma', 2)
    w_a = G.addVarNode('wa', 2)
    m_b = G.addVarNode('mb', 2)
    w_b = G.addVarNode('wb', 2)
    n_ab = G.addVarNode('nab', 2)
    n_ba = G.addVarNode('nba', 2)
    n_aa = G.addVarNode('naa', 2)
    n_bb = G.addVarNode('nbb', 2)

    pot_ma_wa = np.array([[4, 4], [1, 4]])
    pot_mb_wb = np.array([[4, 4], [1, 4]])
    pot_wa_nab_wb = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
    pot_wb_nba_wa = np.array([[[4, 4], [4, 4]], [[4, 4], [1, 4]]])
    pot_wa_naa = np.array([[4, 4], [4, 4]])
    pot_wb_nbb = np.array([[4, 4], [4, 4]])

    G.addFacNode(pot_ma_wa, m_a, w_a)
    G.addFacNode(pot_mb_wb, m_b, w_b)
    G.addFacNode(pot_wa_nab_wb, w_a, n_ab, w_b)
    G.addFacNode(pot_wb_nba_wa, w_b, n_ba, w_a)
    G.addFacNode(pot_wa_naa, w_a, n_aa)
    G.addFacNode(pot_wb_nbb, w_b, n_bb)

    return G

def testToyGraph():

    G = makeToyGraph()
    G.var['ma'].condition(1)
    marg = G.marginals()

    # check the marginals
    mb_marg = marg['mb']
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
    wa_marg = marg['wa']
    print("Walks(Ali) marginals = ",wa_marg)
    wb_marg = marg['wb']
    print("Walks(Bob) marginals = ",wb_marg)
    print('\n')


    brute = G.bruteForce()
    wbbf = G.marginalizeBrute(brute, 'wb')
    print("Walks(Bob) BRUTE_FORCE marginals = ",wbbf,"\n")

# standard run of test cases
testToyGraph()
