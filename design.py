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

import unittest
import collections
import itertools

from numpy import testing
import numpy as np


def _check_codons(codons_in):
    """
    This function is on the front lines of the API.
    As for all functions in this module, speed does not matter.
    It does not make any attempt to process the codons.
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
    This function is on the front lines of the API.
    As for all functions in this module, speed does not matter.
    It does not make any attempt to process the amino acids.
    This function is not too picky about its input.
    @param aminos_in: sequence of amino acids as case insensitive strings
    @return: None
    """
    # just check that the input is a sequence of lowercase-able elements
    aminos = [a.lower() for a in aminos_in]


def _check_codons_aminos(codons_in, aminos_in):
    """
    This function is on the front lines of the API.
    As for all functions in this module, speed does not matter.
    It does not make any attempt to process the codons.
    @param codons_in: sequence of codons as case insensitive strings
    @param aminos_in: sequence of aminos as case insensitive strings
    @return: None
    """
    _check_codons(codons_in)
    _check_aminos(aminos_in)
    if np.shape(codons_in) != np.shape(aminos_in):
        raise ValueError


def get_compo(codons_in):
    """
    Get the nucleotide compositions of the amino acids.
    @param codons_in: sequence of codons as case insensitive strings
    """
    _check_codons(codons_in)
    codons = [c.lower() for c in codons_in]
    ncodons = len(codons)
    compo = np.empty((ncodons, 4))
    for i, c in enumerate(codons):
        for j, nt in enumerate('acgt'):
            compo[i, j] = c.count(nt)
    return compo

def get_hdist(codons_in):
    """
    Get the hamming distances between codons.
    @return: ndarray of shape (ncodons, ncodons) with counts
    """
    _check_codons(codons_in)
    codons = [c.lower() for c in codons_in]
    ncodons = len(codons)
    h = np.empty((ncodons, ncodons), dtype=int)
    for i, ci in enumerate(codons):
        for j, cj in enumerate(codons):
            h[i, j] = np.sum(1 for k in range(3) if ci[k] != cj[k])
    return h

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


class TestMe(testing.TestCase):

    def test_check_codons(self):

        # these codons are valid so no error should be raised
        codons = ['acg', 'Act', 'AAA']
        _check_codons(codons)

        # invalid letter in a codon
        codons = ['acg', 'acx', 'aaa']
        testing.assert_raises(ValueError, _check_codons, codons)

        # codon is not the right length
        codons = ['acg', 'ac', 'aaa']
        testing.assert_raises(ValueError, _check_codons, codons)

        # a codon is repeated
        codons = ['acg', 'act', 'aaa', 'act']
        testing.assert_raises(ValueError, _check_codons, codons)

        # a codon is repeated with case insensitivity
        codons = ['acg', 'AcT', 'aaa', 'act']
        testing.assert_raises(ValueError, _check_codons, codons)

    def test_hdist(self):
        codons = ['acg', 'act', 'aaa']
        observed = get_hdist(codons)
        expected = np.array([
            [0, 1, 2],
            [1, 0, 2],
            [2, 2, 0],
            ], dtype=int)
        testing.assert_array_equal(observed, expected)

    def test_nonsyn(self):

        # some codon pairs are separated by more than one nucleotide change
        codons = ['acg', 'act', 'aaa']
        aminos = ['x', 'y', 'z']
        observed = get_nonsyn(codons, aminos)
        expected = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
            ], dtype=int)
        testing.assert_array_equal(observed, expected)

        # some amino acids are the same
        codons = ['aaa', 'aac', 'aat']
        aminos = ['x', 'y', 'y']
        observed = get_nonsyn(codons, aminos)
        expected = np.array([
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
            ], dtype=int)
        testing.assert_array_equal(observed, expected)

    def test_transitions_transversions_hamming(self):
        codons = list(''.join(x) for x in itertools.product('acgt', repeat=3))
        hdist = get_hdist(codons)
        compo = get_compo(codons)
        sinks = get_nt_sinks(codons, compo=compo, hdist=hdist)
        ts = get_nt_transitions(codons, sinks=sinks)
        tv = get_nt_transversions(codons, sinks=sinks)
        observed = ts + tv
        expected = np.array(np.equal(hdist, 1), dtype=int)
        testing.assert_array_equal(observed, expected)


if __name__ == '__main__':
    testing.run_module_suite()

