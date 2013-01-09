"""
Compute likelihoods using dynamic programming on phylogenetic trees.

This module is concerned only with single alignment columns.
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
.
Some conventions in this module were chosen for consistency with
other modules.
For example it is not so important that these single-pattern functions
return log likelihood instead of plain likelihood,
but when more patterns are involved it becomes easier to work with logs,
so these function signatures were chosen with consistency with these
other functions in mind.
Also the function argument named 'pattern' may seem to use a strange
convention, but it makes the interoperability nicer when the pattern
is allowed to be a row of a numpy ndarray.
Similarly the requirements of vertices and states to be represented
each by an integer in a range starting with zero may seem unnecessarily
restrictive, but it is more convenient for interoperability.
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

def brute(ov, v_to_children, pattern, de_to_P, root_prior):
    """
    This function is only for testing and documentation.
    The P matrices and the root prior may be algopy objects.
    @param ov: ordered vertices with child vertices before parent vertices
    @param v_to_children: map from a vertex to a sequence of child vertices
    @param pattern: an array that maps vertex to state, or to -1 if internal
    @param de_to_P: map from a directed edge to a transition matrix
    @param root_prior: equilibrium distribution at the root
    @return: log likelihood
    """
    nvertices = len(pattern)
    nstates = len(root_prior)
    root = ov[-1]
    v_unknowns = [v for v, state in enumerate(pattern) if state == -1]
    n_unknowns = len(v_unknowns)

    # Construct the set of directed edges on the tree.
    des = set((p, c) for p, cs in v_to_children.items() for c in cs)

    # Compute the likelihood by directly summing over all possibilities.
    likelihood = 0
    for assignment in itertools.product(range(nstates), repeat=n_unknowns):

        # Fill in the state assignments for all vertices.
        augmented_pattern = np.array(pattern)
        for v, state in zip(v_unknowns, assignment):
            augmented_pattern[v] = state

        # Add to the log likelihood.
        edge_prob = 1.0
        for p, c in des:
            p_state = augmented_pattern[p]
            c_state = augmented_pattern[c]
            edge_prob *= de_to_P[p, c][p_state, c_state]
        likelihood += root_prior[augmented_pattern[root]] * edge_prob

    # Return the log likelihood.
    return algopy.log(likelihood)


def fels(ov, v_to_children, pattern, de_to_P, root_prior):
    """
    The P matrices and the root prior may be algopy objects.
    @param ov: ordered vertices with child vertices before parent vertices
    @param v_to_children: map from a vertex to a sequence of child vertices
    @param pattern: an array that maps vertex to state, or to -1 if internal
    @param de_to_P: map from a directed edge to a transition matrix
    @param root_prior: equilibrium distribution at the root
    @return: log likelihood
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
        state = pattern[v]
        if state >= 0:
            for s in range(nstates):
                if s != state:
                    likelihoods[v, s] = 0

    # Get the log likelihood by summing over equilibrium states at the root.
    return algopy.log(algopy.dot(root_prior, likelihoods[root]))


def get_jc_rate_matrix():
    """
    This is only for testing.
    It returns a continuous-time Jukes-Cantor rate matrix
    normalized to one expected substitution per time unit.
    """
    nstates = 4
    pre_Q_jc = np.ones((nstates, nstates), dtype=float)
    Q_jc = pre_Q_jc - np.diag(np.sum(pre_Q_jc, axis=1))
    return Q_jc * (4.0 / 3.0)


class TestLikelihood(testing.TestCase):

    def test_likelihood_internal_root(self):
        nstates = 4
        ov = (3, 2, 1, 0)
        pattern = np.array([-1, 0, 0, 1])
        #v_to_state = {0 : None, 1 : None, 2 : None, 3 : None}
        v_to_children = {0 : [1, 2, 3]}
        Q_jc = get_jc_rate_matrix()
        de_to_P = {
                (0, 1) : scipy.linalg.expm(1 * Q_jc),
                (0, 2) : scipy.linalg.expm(2 * Q_jc),
                (0, 3) : scipy.linalg.expm(3 * Q_jc),
                }
        root_prior = np.ones(nstates) / float(nstates)
        #root_prior = np.array([2, 1, 0, 0]) / float(3)
        ll = brute(ov, v_to_children, pattern, de_to_P, root_prior)
        print ll
        ll = fels(ov, v_to_children, pattern, de_to_P, root_prior)
        print ll
        testing.assert_allclose(ll, -1)

    def test_likelihood_leaf_root(self):
        nstates = 4
        ov = (3, 2, 0, 1)
        pattern = np.array([-1, 0, 0, 1])
        v_to_children = {1: [0], 0 : [2, 3]}
        Q_jc = get_jc_rate_matrix()
        de_to_P = {
                (1, 0) : scipy.linalg.expm(1 * Q_jc),
                (0, 2) : scipy.linalg.expm(2 * Q_jc),
                (0, 3) : scipy.linalg.expm(3 * Q_jc),
                }
        root_prior = np.ones(nstates) / float(nstates)
        #root_prior = np.array([2, 1, 0, 0]) / float(3)
        ll = brute(ov, v_to_children, pattern, de_to_P, root_prior)
        print ll
        ll = fels(ov, v_to_children, pattern, de_to_P, root_prior)
        print ll
        testing.assert_allclose(ll, -1)


if __name__ == '__main__':
    testing.run_module_suite()

