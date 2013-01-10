"""
Speed up log likelihood calculations.

The implementation is in Cython for speed
and uses python numpy arrays for speed and convenience.
For compilation instructions see
http://docs.cython.org/src/reference/compilation.html
For example:
$ cython -a llcore.pyx
$ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
      -I/usr/include/python2.7 -o llcore.so llcore.c
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport log

np.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def align_fels(
        np.ndarray[np.int_t, ndim=1] OV,
        np.ndarray[np.int_t, ndim=2] DE,
        np.ndarray[np.int_t, ndim=2] patterns,
        np.ndarray[np.int_t, ndim=1] pattern_mults,
        np.ndarray[np.float64_t, ndim=3] multi_P,
        np.ndarray[np.float64_t, ndim=1] root_prior,
        ):
    """
    @param OV: ordered vertices with child vertices before parent vertices
    @param DE: array of directed edges
    @param patterns: each pattern maps each vertex to a state or to -1
    @param pattern_mults: pattern multiplicites
    @param multi_P: a transition matrix for each directed edge
    @param root_prior: equilibrium distribution at the root
    @return: log likelihood
    """
    cdef int nvertices = OV.shape[0]
    cdef int nstates = root_prior.shape[0]
    cdef int npatterns = pattern_mults.shape[0]
    cdef double pattern_ll
    cdef double total_ll

    # allocate the ndarray that maps vertices to subtree likelihoods
    cdef np.ndarray[np.float64_t, ndim=2] likelihoods = np.empty(
            (nvertices, nstates), dtype=float)

    # sum over all of the patterns
    total_ll = 0.0
    for pattern_index in range(npatterns):
        pattern_ll = site_fels_with_workspace(
            OV, DE, patterns[pattern_index], multi_P, root_prior, likelihoods)
        total_ll += pattern_ll * pattern_mults[pattern_index]
    return total_ll


@cython.boundscheck(False)
@cython.wraparound(False)
def site_fels(
        np.ndarray[np.int_t, ndim=1] OV,
        np.ndarray[np.int_t, ndim=2] DE,
        np.ndarray[np.int_t, ndim=1] pattern,
        np.ndarray[np.float64_t, ndim=3] multi_P,
        np.ndarray[np.float64_t, ndim=1] root_prior,
        ):
    """
    @param OV: ordered vertices with child vertices before parent vertices
    @param DE: array of directed edges
    @param pattern: maps vertex to state, or to -1 if internal
    @param multi_P: a transition matrix for each directed edge
    @param root_prior: equilibrium distribution at the root
    @return: log likelihood
    """

    # init counts and indices
    cdef int nvertices = OV.shape[0]
    cdef int nstates = root_prior.shape[0]

    # initialize the map from vertices to subtree likelihoods
    cdef np.ndarray[np.float64_t, ndim=2] likelihoods = np.empty(
            (nvertices, nstates), dtype=float)

    # return the log likelihood
    return site_fels_with_workspace(
            OV, DE, pattern, multi_P, root_prior, likelihoods)


@cython.boundscheck(False)
@cython.wraparound(False)
def site_fels_with_workspace(
        np.ndarray[np.int_t, ndim=1] OV,
        np.ndarray[np.int_t, ndim=2] DE,
        np.ndarray[np.int_t, ndim=1] pattern,
        np.ndarray[np.float64_t, ndim=3] multi_P,
        np.ndarray[np.float64_t, ndim=1] root_prior,
        np.ndarray[np.float64_t, ndim=2] likelihoods,
        ):
    """
    @param OV: ordered vertices with child vertices before parent vertices
    @param DE: array of directed edges
    @param pattern: maps vertex to state, or to -1 if internal
    @param multi_P: a transition matrix for each directed edge
    @param root_prior: equilibrium distribution at the root
    @param likelihoods: an (nvertices, nstates) array
    @return: log likelihood
    """

    # init counts and indices
    cdef int nvertices = OV.shape[0]
    cdef int nstates = root_prior.shape[0]
    cdef int nedges = DE.shape[0]
    cdef int root = OV[nvertices - 1]

    # declare more variables
    cdef int parent_vertex
    cdef int child_vertex
    cdef int parent_state
    cdef int child_state
    cdef int pattern_state
    cdef double log_likelihood

    # utility variables for computing a dot product manually
    cdef int i
    cdef double accum
    cdef double a_i, b_i

    # Compute the subtree likelihoods using dynamic programming.
    for parent_vertex_index in range(nvertices):
        parent_vertex = OV[parent_vertex_index]
        pattern_state = pattern[parent_vertex]
        for parent_state in range(nstates):
            if pattern_state != -1 and parent_state != pattern_state:
                likelihoods[parent_vertex, parent_state] = 0.0
            else:
                likelihoods[parent_vertex, parent_state] = 1.0
                for edge_index in range(nedges):
                    if DE[edge_index, 0] == parent_vertex:
                        child_vertex = DE[edge_index, 1]
                        accum = 0.0
                        for i in range(nstates):
                            a_i = multi_P[edge_index, parent_state, i]
                            b_i = likelihoods[child_vertex, i]
                            accum += a_i * b_i
                        likelihoods[parent_vertex, parent_state] *= accum

    # Get the log likelihood by summing over equilibrium states at the root.
    accum = 0.0
    for i in range(nstates):
        a_i = root_prior[i]
        b_i = likelihoods[root, i]
        accum += a_i * b_i
    log_likelihood = log(accum)
    return log_likelihood

