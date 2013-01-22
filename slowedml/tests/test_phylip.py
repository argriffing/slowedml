
from StringIO import StringIO
import unittest

from slowedml.phylip import *


class TestPhylip(unittest.TestCase):
    
    def test_read_interleaved_codon_alignment(self):
        fin = StringIO(''
                '2 6\n'
                'foo ACG\n'
                'bar GCA\n'
                'AAA\n'
                'CCC\n'
                )
        observed = list(read_interleaved_codon_alignment(fin))
        expected = [
                ('foo', 'bar'),
                ('ACG', 'GCA'),
                ('AAA', 'CCC')]
        self.assertEqual(observed, expected)


if __name__ == '__main__':
    unittest.main()

