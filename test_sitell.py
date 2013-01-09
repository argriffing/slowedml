"""
Test site likelihood implementation of the Felsenstein pruning algorithm.
"""

import numpy as np
from numpy import testing
import scipy.linalg

import sitell


def get_jc_rate_matrix():
    """
    This is only for testing.
    It returns a continuous-time Jukes-Cantor rate matrix
    normalized to one expected substitution per time unit.
    """
    nstates = 4
    pre_Q_jc = np.ones((nstates, nstates), dtype=float)
    Q_jc = pre_Q_jc - np.diag(np.sum(pre_Q_jc, axis=1))
    return Q_jc * (1.0 / 3.0)


class Test_SiteLikelihood(testing.TestCase):

    def test_likelihood_internal_root(self):
        nstates = 4
        ov = (3, 2, 1, 0)
        pattern = np.array([-1, 0, 0, 1])
        #v_to_state = {0 : None, 1 : None, 2 : None, 3 : None}
        v_to_children = {0 : [1, 2, 3]}
        Q_jc = get_jc_rate_matrix()
        de_to_P = {
                (0, 1) : scipy.linalg.expm(0.1 * Q_jc),
                (0, 2) : scipy.linalg.expm(0.2 * Q_jc),
                (0, 3) : scipy.linalg.expm(0.3 * Q_jc),
                }
        root_prior = np.ones(nstates) / float(nstates)
        expected_ll = -4.14671850148
        ll = sitell.brute(ov, v_to_children, pattern, de_to_P, root_prior)
        testing.assert_allclose(ll, expected_ll)
        ll = sitell.fels(ov, v_to_children, pattern, de_to_P, root_prior)
        testing.assert_allclose(ll, expected_ll)

    def test_likelihood_leaf_root(self):
        nstates = 4
        ov = (3, 2, 0, 1)
        pattern = np.array([-1, 0, 0, 1])
        v_to_children = {1: [0], 0 : [2, 3]}
        Q_jc = get_jc_rate_matrix()
        de_to_P = {
                (1, 0) : scipy.linalg.expm(0.1 * Q_jc),
                (0, 2) : scipy.linalg.expm(0.2 * Q_jc),
                (0, 3) : scipy.linalg.expm(0.3 * Q_jc),
                }
        root_prior = np.ones(nstates) / float(nstates)
        expected_ll = -4.14671850148
        ll = sitell.brute(ov, v_to_children, pattern, de_to_P, root_prior)
        testing.assert_allclose(ll, expected_ll)
        ll = sitell.fels(ov, v_to_children, pattern, de_to_P, root_prior)
        testing.assert_allclose(ll, expected_ll)


if __name__ == '__main__':
    testing.run_module_suite()

