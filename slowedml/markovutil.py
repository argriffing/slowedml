"""
Miscellaneous functions related to continuous time Markov processes.
"""

import algopy


def pre_Q_to_Q(pre_Q, stationary_distn, target_expected_rate):
    """
    Return a matrix with a different diagonal and a different scaling.
    """
    unscaled_Q = pre_Q - algopy.diag(algopy.sum(pre_Q, axis=1))
    r = -algopy.dot(algopy.diag(unscaled_Q), stationary_distn)
    Q = (target_expected_rate / r) * unscaled_Q
    return Q

