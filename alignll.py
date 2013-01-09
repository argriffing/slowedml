"""
Likelihoods involving alignment column patterns and multiplicities.

Each vertex is assumed to be an integer in a range starting with zero.
The patterns matrix has a row for each pattern and a column for each vertex.
Vertices that do not correspond to a sequence in the alignment will
have value -1 in the pattern matrix.
Other vertices will have a value that correspond to a state,
for example a nucleotide or a codon or whatever.
"""

import numpy as np
import algopy

import sitell

def _ll_helper(
        ov, v_to_children, de_to_P, root_prior,
        patterns, pat_mults,
        fn,
        ):
    """
    This is purely a helper function.
    It should not be called outside of its own module.
    The P matrices and the root prior may be algopy objects.
    @param ov: ordered vertices with child vertices before parent vertices
    @param v_to_children: map from a vertex to a sequence of child vertices
    @param de_to_P: map from a directed edge to a transition matrix
    @param root_prior: equilibrium distribution at the root
    @param patterns: each pattern assigns a state to each leaf
    @param pat_mults: a multiplicity for each pattern
    @param fn: a per-pattern log likelihood evaluation function
    @return: log likelihood
    """
    npatterns = patterns.shape[0]
    lls = algopy.zeros(npatterns, dtype=root_prior)
    for i in range(npatterns):
        lls[i] = fn(ov, v_to_children, patterns[i], de_to_p, root_prior)
    return algopy.dot(lls, pat_mults)

def brute(
        ov, v_to_children, de_to_P, root_prior,
        patterns, pat_mults,
        ):
    """
    This function is only for testing and documentation.
    The P matrices and the root prior may be algopy objects.
    @param ov: ordered vertices with child vertices before parent vertices
    @param v_to_children: map from a vertex to a sequence of child vertices
    @param de_to_P: map from a directed edge to a transition matrix
    @param root_prior: equilibrium distribution at the root
    @param patterns: each pattern assigns a state to each leaf
    @param pat_mults: a multiplicity for each pattern
    @return: log likelihood
    """
    return _ll_helper(
            ov, v_to_children, de_to_P, root_prior,
            patterns, pat_mults,
            sitelike.brute,
            )

def fels(
        ov, v_to_children, de_to_P, root_prior,
        patterns, pat_mults,
        ):
    """
    The P matrices and the root prior may be algopy objects.
    @param ov: ordered vertices with child vertices before parent vertices
    @param v_to_children: map from a vertex to a sequence of child vertices
    @param de_to_P: map from a directed edge to a transition matrix
    @param root_prior: equilibrium distribution at the root
    @param patterns: each pattern assigns a state to each leaf
    @param pat_mults: a multiplicity for each pattern
    @return: log likelihood
    """
    return _ll_helper(
            ov, v_to_children, de_to_P, root_prior,
            patterns, pat_mults,
            sitelike.fels,
            )

