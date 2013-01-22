"""
Construct design matrices from primitive information about the genetic code.

The idea is that other scripts will read the genetic code description files
themselves, and then they will minimally parse these files well enough
to feed the genetic code description to this module.
This module then uses the genetic code description to construct
medium-sized (e.g. 64x64) numpy ndarrays which are passed back to the script.
The script will then use these ndarrays to do things like
evaluate log likelihoods.
Note that nothing in this module should be speed-limiting,
so the functions do not have to be optimized for efficiency.
The API is function-oriented instead of object-oriented,
and the client describes the genetic code by passing two sequences.
The first sequence gives the ordered codons,
and the second sequence gives the corresponding amino acids.
The elements of the sequences are assumed to be case-insensitive strings,
and in particular the codons are assumed to be triples
of case insensitive letters in {a, c, g, t, A, C, G, T}.
For the purposes of this module, the only thing that matters about the
amino acids is which ones are the same as each other,
except for stop codon which are represented by the
case insensitive amino acid string 'stop'.
"""

import collections
import itertools

import numpy as np


##############################################################################
# These functions do some input validation.

def _check_codons(codons_in):
    """
    @param codons_in: sequence of codons as case insensitive strings
    @return: None
    """
    codons = [c.lower() for c in codons_in]
    # check for invalid codons
    valid_codons = set(''.join(x) for x in itertools.product('acgt', repeat=3))
    invalid_codons = set(codons) - set(valid_codons)
    if invalid_codons:
        raise ValueError('invalid codons: %s' % str(invalid_codons))
    # check for repeated codons
    codon_counts = collections.Counter(codons)
    repeated_codons = [c for c, n in codon_counts.items() if n > 1]
    if repeated_codons:
        raise ValueError('repeated codons: %s' % str(repeated_codons))

def _check_aminos(aminos_in):
    """
    @param aminos_in: sequence of amino acids as case insensitive strings
    @return: None
    """
    # just check that the input is a sequence of lowercase-able elements
    aminos = [a.lower() for a in aminos_in]

def _check_codons_aminos(codons_in, aminos_in):
    """
    @param codons_in: sequence of codons as case insensitive strings
    @param aminos_in: sequence of aminos as case insensitive strings
    @return: None
    """
    _check_codons(codons_in)
    _check_aminos(aminos_in)
    if np.shape(codons_in) != np.shape(aminos_in):
        raise ValueError


##############################################################################
# These functions convert the genetic code into ndarrays of integers.
# The inputs describe how the codons should be ordered,
# and which pairs of codons are translated into the same amino acid.

def get_full_compo(codons_in):
    """
    @return: a binary ndarray of shape (ncodons, 3, 4)
    """
    _check_codons(codons_in)
    codons = [c.lower() for c in codons_in]
    ncodons = len(codons)
    full_compo = np.zeros((ncodons, 3, 4), dtype=int)
    for i, c in enumerate(codons):
        for j in range(3):
            for k, nt in enumerate('acgt'):
                if c[j] == nt:
                    full_compo[i, j, k] = 1
    return full_compo

def get_compo(codons_in, full_compo=None):
    """
    Get the nucleotide compositions of the amino acids.
    @param codons_in: sequence of codons as case insensitive strings
    """
    _check_codons(codons_in)
    if full_compo is None:
        full_compo = get_full_compo(codons_in)
    return np.sum(full_compo, axis=1)

def get_hdist(codons_in):
    """
    Get the hamming distances between codons.
    @return: ndarray of shape (ncodons, ncodons) with counts
    """
    _check_codons(codons_in)
    codons = [c.lower() for c in codons_in]
    ncodons = len(codons)
    hdist = np.empty((ncodons, ncodons), dtype=int)
    for i, ci in enumerate(codons):
        for j, cj in enumerate(codons):
            hdist[i, j] = np.sum(1 for k in range(3) if ci[k] != cj[k])
    return hdist

def get_adjacency(codons_in, hdist=None):
    """
    The returned binary matrix specifies which codons are 1 nucleotide apart.
    @return: ndarray of shape (ncodons, ncodons) with counts
    """
    _check_codons(codons_in)
    if hdist is None:
        hdist = get_hdist(codons_in)
    return np.equal(hdist, 1).astype(int)

def get_nt_sinks(codons_in, compo=None, hdist=None):
    """
    Returns an ndim-3 mask corresponding to sinks of single nucleotide changes.
    The returned mask[i, j, k] has value 1 if codon_j -> codon_k
    is a single nucleotide change to nucleotide i, and has value 0 otherwise.
    @return: an ndim-3 mask
    """
    _check_codons(codons_in)
    codons = [c.lower() for c in codons_in]
    ncodons = len(codons)
    if hdist is None:
        hdist = get_hdist(codons_in)
    if compo is None:
        compo = get_compo(codons_in)
    sink_mask = np.zeros((4, ncodons, ncodons), dtype=int)
    for i, nt in enumerate('acgt'):
        for j, cj in enumerate(codons):
            for k, ck in enumerate(codons):
                if hdist[j, k] == 1:
                    if compo[k, i] - compo[j, i] == 1:
                        sink_mask[i, j, k] = 1
    return sink_mask

def get_nt_transitions(codons_in, sinks=None):
    """
    The returned binary matrix is 1 when a codon change is a transition.
    Here, a transition is a technical type of single nucleotide change.
    @return: an ndarray mask of shape (ncodons, ncodons)
    """
    _check_codons(codons_in)
    if sinks is None:
        sinks = get_nt_sinks(codons_in)
    a, c, g, t = sinks
    forward = a.T * t + c.T * g
    return forward + forward.T

def get_nt_transversions(codons_in, sinks=None):
    """
    The returned binary matrix is 1 when a codon change is a transversion.
    Here, a transversion is a technical type of single nucleotide change.
    @return: an ndarray mask of shape (ncodons, ncodons)
    """
    _check_codons(codons_in)
    if sinks is None:
        sinks = get_nt_sinks(codons_in)
    a, c, g, t = sinks
    forward = a.T * c + a.T * g + t.T * c + t.T * g
    return forward + forward.T

def get_nonsyn(codons_in, aminos_in, hdist=None):
    """
    Entries should be 1 for non-synonymous single nucleotide changes.
    @param codons_in: sequence of codons as case insensitive strings
    @param aminos_in: sequence of aminos as case insensitive strings
    @return: binary mask of shape (ncodons, ncodons)
    """
    _check_codons_aminos(codons_in, aminos_in)
    codons = [c.lower() for c in codons_in]
    aminos = [a.lower() for a in aminos_in]
    ncodons = len(codons)
    if hdist is None:
        hdist = get_hdist(codons_in)
    nonsyn = np.zeros((ncodons, ncodons), dtype=int)
    for i, ci in enumerate(codons):
        for j, cj in enumerate(codons):
            if hdist[i, j] == 1 and aminos[i] != aminos[j]:
                nonsyn[i, j] = 1
    return nonsyn

def get_syn(codons_in, aminos_in, hdist=None):
    _check_codons_aminos(codons_in, aminos_in)
    adjacency = get_adjacency(codons_in, hdist=hdist)
    return adjacency - get_nonsyn(codons_in, aminos_in, hdist=hdist)


##############################################################################
# These functions convert the alignment data into ndarrays of integers.

def get_single_site_pattern_array(codons_in, codon_alignment_column_in):
    """
    The first argument defines the order of the codon states.
    @param codons_in: sequence of codons as case insensitive strings
    @param codon_alignment_column_in: a sequence of codons
    @return: a one dimensional state array of length ntaxa
    """
    _check_codons(codons_in)
    codons = [c.lower() for c in codons_in]
    c_to_i = dict((c, i) for i, c in enumerate(codons))
    column = [c.lower() for c in codon_alignment_column_in]
    return np.array([c_to_i[c] for c in column], dtype=int)

def get_pattern_array(codons_in, codon_alignment_columns_in):
    """
    The first argument defines the order of the codon states.
    @param codons_in: sequence of codons as case insensitive strings
    @param codon_alignment_columns_in: a sequence of codon sequences
    @return: an ndarray of shape (nsites, ntaxa)
    """
    _check_codons(codons_in)
    codons = [c.lower() for c in codons_in]
    c_to_i = dict((c, i) for i, c in enumerate(codons))
    nsites = len(codon_alignment_columns_in)
    ntaxa = None
    M = None
    for i, column_in in enumerate(codon_alignment_columns_in):
        column = [c.lower() for c in column_in]
        if i == 0:
            ntaxa = len(column)
            M = np.empty((nsites, ntaxa), dtype=int)
        if len(column) != ntaxa:
            raise ValueError(
                    'each column should have the same number of elements')
        for j, c in enumerate(column):
            M[i, j] = c_to_i[c]
    return M
