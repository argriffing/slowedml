#!/usr/bin/env python

import argparse

from slowedml import fileutil


def main(taxa, fout):
    print >> fout, '(' + ', '.join(taxa) + ');'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-o', default='-',
            help='output a newick tree (default stdout)')
    parser.add_argument('taxa', nargs='+',
            help='taxa to include in the output')
    args = parser.parse_args()
    with fileutil.open_or_stdout(args.o, 'w') as fout:
        main(args.taxa, fout)
