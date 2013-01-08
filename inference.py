"""
Compute likelihoods using dynamic programming on phylogenetic trees.

Dynamic programming on tree structured graphs in the context of phylogenetics
was introduced by Joseph Felsenstein,
and this concept is often is associated with the words "pruning" or "peeling"
especially when it involves the evaluation of a likelihood function.
In this module, vertices are unique hashable things like python integers.
States are integers in some range starting with zero.
Directed edges are pairs of vertices.
Matrices are numpy ndarrays or algopy objects.
The algopy presence is for computing the hessian
through a Felsenstein pruning algorithm,
and also for providing local curvature information about the log likelihood
function for nonlinear optimization.
"""

import itertools
import math

import numpy as np
from numpy import testing
import scipy.linalg
import algopy

# Use the following notation.
# de: directed edge represented by a tuple of two integers
# ue: undirected edge represented by a frozenset of two integers
# ov: ordered sequence of vertices
# uv: unordered set of vertices
# v_to_children: map from a vertex to a collection of child vertices
# P: transition matrix

def get_likelihood_brute(ov, v_to_children, v_to_state, de_to_P, root_prior):
    """
    This function is only for testing and documentation.
    The P matrices and the root prior may be algopy objects.
    @param ov: ordered vertices with child vertices before parent vertices
    @param v_to_children: map from a vertex to a sequence of child vertices
    @param v_to_state: map from a vertex to None or to a known state
    @param de_to_P: map from a directed edge to a transition matrix
    @param root_prior: equilibrium distribution at the root
    """
    nstates = len(root_prior)
    root = ov[-1]
    v_unknowns = [v for v, state in v_to_state.items() if state is None]
    n_unknowns = len(v_unknowns)

    # Construct the set of directed edges on the tree.
    des = set((p, c) for p, cs in v_to_children.items() for c in cs)

    # Compute the likelihood by directly summing over all possibilities.
    likelihood = 0
    for assignment in itertools.product(range(nstates), repeat=n_unknowns):

        # Fill in the state assignments for all vertices.
        v_to_s = dict(v_to_state)
        v_to_s.update(dict(zip(v_unknowns, assignment)))

        # Add to the log likelihood.
        # This could use algopy.prod but it is a little unfinished.
        edge_prob = 1.0
        for p, c in des:
            edge_prob *= de_to_P[p, c][v_to_s[p], v_to_s[c]]

        #edge_prob = algopy.prod([
            #de_to_P[p, c][v_to_s[p], v_to_s[c]] for p, c in des])

        likelihood += root_prior[v_to_s[root]] * edge_prob

    # Return the likelihood.
    return likelihood


def get_likelihood_fels(ov, v_to_children, v_to_state, de_to_P, root_prior):
    """
    The P matrices and the root prior may be algopy objects.
    @param ov: ordered vertices with child vertices before parent vertices
    @param v_to_children: map from a vertex to a sequence of child vertices
    @param v_to_state: map from a vertex to None or to a known state
    @param de_to_P: map from a directed edge to a transition matrix
    @param root_prior: equilibrium distribution at the root
    """
    nvertices = len(ov)
    nstates = len(root_prior)
    states = range(nstates)
    root = ov[-1]

    # Initialize the map from vertices to subtree likelihoods.
    likelihoods = algopy.ones((nvertices, nstates), dtype=root_prior)

    # Compute the subtree likelihoods using dynamic programming.
    for v in ov:
        for pstate in range(nstates):
            for c in v_to_children.get(v, []):
                P = de_to_P[v, c]
                likelihoods[v, pstate] *= algopy.dot(P[pstate], likelihoods[c])
        state = v_to_state[v]
        if state is not None:
            for s in range(nstates):
                if s != state:
                    likelihoods[v, s] = 0

    # Get the likelihood by summing over equilibrium states at the root.
    return algopy.dot(root_prior, likelihoods[root])


def get_jc_rate_matrix():
    """
    This is only for testing.
    """
    nstates = 4
    pre_Q_jc = np.ones((nstates, nstates), dtype=float)
    Q_jc = pre_Q_jc - np.diag(np.sum(pre_Q_jc, axis=1))
    return Q_jc


class TestLikelihood(testing.TestCase):

    def test_likelihood_internal_root(self):
        nstates = 4
        ov = (3, 2, 1, 0)
        v_to_state = {0 : None, 1 : 0, 2 : 0, 3 : 1}
        #v_to_state = {0 : None, 1 : None, 2 : None, 3 : None}
        v_to_children = {0 : [1, 2, 3]}
        #Q_jc = (4.0 / 3.0) * get_jc_rate_matrix()
        #Q_jc = get_jc_rate_matrix()
        Q_jc = (3.0 / 4.0) * get_jc_rate_matrix()
        #Q_jc = 0.25 * get_jc_rate_matrix()
        de_to_P = {
                (0, 1) : scipy.linalg.expm(1 * Q_jc),
                (0, 2) : scipy.linalg.expm(2 * Q_jc),
                (0, 3) : scipy.linalg.expm(3 * Q_jc),
                }
        #root_prior = np.ones(nstates) / float(nstates)
        root_prior = np.array([2, 1, 0, 0]) / float(3)
        likelihood = get_likelihood_brute(
                ov, v_to_children, v_to_state, de_to_P, root_prior)
        print math.log(likelihood)
        likelihood = get_likelihood_fels(
                ov, v_to_children, v_to_state, de_to_P, root_prior)
        print math.log(likelihood)
        testing.assert_allclose(likelihood, -1)

    def test_likelihood_leaf_root(self):
        nstates = 4
        ov = (3, 2, 0, 1)
        v_to_state = {0 : None, 1 : 0, 2 : 0, 3 : 1}
        v_to_children = {1: [0], 0 : [2, 3]}
        #Q_jc = (4.0 / 3.0) * get_jc_rate_matrix()
        Q_jc = get_jc_rate_matrix()
        #Q_jc = 0.25 * get_jc_rate_matrix()
        de_to_P = {
                (1, 0) : scipy.linalg.expm(1 * Q_jc),
                (0, 2) : scipy.linalg.expm(2 * Q_jc),
                (0, 3) : scipy.linalg.expm(3 * Q_jc),
                }
        #root_prior = np.ones(nstates) / float(nstates)
        root_prior = np.array([2, 1, 0, 0]) / float(3)
        likelihood = get_likelihood_brute(
                ov, v_to_children, v_to_state, de_to_P, root_prior)
        print math.log(likelihood)
        likelihood = get_likelihood_fels(
                ov, v_to_children, v_to_state, de_to_P, root_prior)
        print math.log(likelihood)
        testing.assert_allclose(likelihood, -1)


if __name__ == '__main__':
    testing.run_module_suite()

