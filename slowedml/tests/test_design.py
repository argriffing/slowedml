"""
One of these tests sets a random number seed to reduce testing randomness.
"""

import random
import itertools

from numpy import testing
import numpy as np

from slowedml.design import *
from slowedml.design import _check_codons
from slowedml.design import _check_aminos
from slowedml.design import _check_codons_aminos


class TestDesign(testing.TestCase):

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

    def test_adjacency_hamming(self):
        codons = list(''.join(x) for x in itertools.product('acgt', repeat=3))
        hdist = get_hdist(codons)
        for h in (None, hdist):
            observed = get_adjacency(codons, hdist=h)
            expected = np.equal(hdist, 1).astype(int)
            testing.assert_array_equal(observed, expected)

    def test_transitions_transversions_hamming(self):
        codons = list(''.join(x) for x in itertools.product('acgt', repeat=3))
        hdist = get_hdist(codons)
        compo = get_compo(codons)
        sinks = get_nt_sinks(codons, compo=compo, hdist=hdist)
        for s in (None, sinks):
            adjacency = get_adjacency(codons, hdist=hdist)
            ts = get_nt_transitions(codons, sinks=s)
            tv = get_nt_transversions(codons, sinks=s)
            testing.assert_array_equal(ts + tv, adjacency)
            testing.assert_array_equal(ts * tv, np.zeros_like(adjacency))

    def test_syn_nonsyn_hamming(self):
        random.seed(0)
        codons = list(''.join(x) for x in itertools.product('acgt', repeat=3))
        aminos = [random.choice('uvwxyz') for c in codons]
        hdist = get_hdist(codons)
        for h in (None, hdist):
            adjacency = get_adjacency(codons, hdist=h)
            syn = get_syn(codons, aminos, hdist=h)
            nonsyn = get_nonsyn(codons, aminos, hdist=h)
            testing.assert_array_equal(syn + nonsyn, adjacency)
            testing.assert_array_equal(syn * nonsyn, np.zeros_like(adjacency))

    def test_single_site_pattern_array(self):
        codons = ['aaa', 'aac', 'aat']
        column = ['AAC', 'aat', 'aaa', 'aaa']
        observed = get_single_site_pattern_array(codons, column)
        expected = np.array([1, 2, 0, 0], dtype=int)
        testing.assert_array_equal(observed, expected)

    def test_pattern_array(self):
        codons = ['aaa', 'aac', 'aat']
        columns = [
                ['AAC', 'aat', 'aaa', 'aaa'],
                ['aat', 'aAa', 'aaa', 'aac']]
        observed = get_pattern_array(codons, columns)
        expected = np.array([
            [1, 2, 0, 0],
            [2, 0, 0, 1],
            ], dtype=int)
        testing.assert_array_equal(observed, expected)


if __name__ == '__main__':
    testing.run_module_suite()

