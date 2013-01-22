"""
Check invariants.
"""

import numpy as np


def assert_valid_distn(distn):
    if np.ndim(distn) != 1:
        raise Exception
    if not np.all(np.greater_equal(distn, 0)):
        raise Exception
    if not np.allclose(np.sum(distn), 1):
        raise Exception
