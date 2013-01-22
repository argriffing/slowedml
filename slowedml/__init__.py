"""
In this package init file we import some modules.
"""

import sitell
import alignll
import design
import pedant
import moretypes
import phylip

# import this package so that nosetests can find our tests
import tests


# testing
from numpy.testing import Tester
test = Tester().test

