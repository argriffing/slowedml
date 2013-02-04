"""
Test a model that is similar to one used by Yang and Nielsen in 2008.
"""

import numpy as np
from numpy import testing
import scipy.linalg

from slowedml import fmutsel


class Test_KimuraCore(testing.TestCase):

    def test_kimura_genic(self):
        nstates = 5
        S = np.random.randn(nstates, nstates)
        kimura_d = 0.0
        H_expected = fmutsel.genic_fixation(S)
        H_observed = fmutsel.unconstrained_recessivity_fixation(kimura_d, S)
        #H_c = fmutsel.unconstrained_recessivity_fixation(kimura_d, 2*S)
        #H_d = fmutsel.unconstrained_recessivity_fixation(kimura_d, 0.5*S)
        #print H_a
        #print H_b
        #print H_c
        #print H_d
        testing.assert_allclose(H_expected, H_observed)
        #raise Exception


if __name__ == '__main__':
    testing.run_module_suite()

