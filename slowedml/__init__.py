"""
In this package init file we import some modules.
"""

import sitell
import alignll
import design
import pedant
import moretypes
import phylip
import algopyboilerplate
import fileutil

# these modules are for Markov models of molecular evolution
import markovutil
import fmutsel
import codon1994

# testing
from numpy.testing import Tester
test = Tester().test

