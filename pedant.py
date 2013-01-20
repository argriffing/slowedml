"""
Validate inputs and check invariants.
"""

import numpy as np


#############################################################################
# These functions are argparse helper types.

def pos_int(x):
    x = int(x)
    if x < 1:
        raise TypeError
    return x

def nonneg_int(x):
    x = int(x)
    if x < 0:
        raise TypeError
    return x


#############################################################################
# These functions are for ndarray invariant validation.

def assert_valid_distn(distn):
    if np.ndim(distn) != 1:
        raise Exception
    if not np.all(np.greater_equal(distn, 0)):
        raise Exception
    if not np.allclose(np.sum(distn), 1):
        raise Exception
