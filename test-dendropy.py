
import sys

import dendropy


def main(fin):
    """
    @param fin: open file for reading
    """
    d = dendropy.DnaCharacterMatrix.get_from_stream(
            fin,
            'phylip',
            datatype='dna',
            interleaved=True,
            strict=True,
            )
    print d.description(3)

if __name__ == '__main__':
    main(sys.stdin)

