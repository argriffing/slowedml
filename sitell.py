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
import algopy


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
    likelihoods = algopy.ones(
            (nvertices, nstates),
            dtype=de_to_P.values()[0],
            )

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

