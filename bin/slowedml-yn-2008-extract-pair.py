#!/usr/bin/env python

"""
Input and output are interleaved Phylip codon sequences.

The output will have only two sequences.
It is meant to be compatible with paml.
"""

import sys
import argparse

from slowedml import fileutil


def gen_paragraphs(fin):
    paragraph = []
    for line in fin:
        line = line.rstrip()
        if not line:
            if paragraph:
                yield paragraph
            paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield paragraph


def main(selected_taxa, fin, fout):
    """
    This is a very simple filter.
    Except for the header,
    the lines written to the output are a subset
    of the lines read from the input.
    @param selected_taxa: write lines corresponding to these taxa
    @param fin: open phylip file for reading
    @param fout: open phylip file for writing
    """

    ntaxa_selected = len(selected_taxa)

    # read the header
    header_row = fin.readline().split()
    if len(header_row) != 3:
        raise Exception(
                'expected the first line to be a header with three elements')
    s_ntaxa, s_nnucs, s_I = header_row
    if s_I != 'I':
        raise Exception(
                "expected the third element of the header to be 'I'")
    ntaxa = int(s_ntaxa)
    nnucs = int(s_nnucs)

    # write the header
    row = (ntaxa_selected, nnucs, 'I')
    print >> fout, ' ' + ' '.join(str(x) for x in row)

    # read the file in paragraph form
    paragraphs = list(gen_paragraphs(fin))

    # get the taxon names from the first paragraph
    name_to_index = {}
    for i, line in enumerate(paragraphs[0]):
        name = line.split()[0]
        name_to_index[name] = i

    # check that the selected taxa are in the file
    missing_taxa = set(selected_taxa) - set(name_to_index)
    if missing_taxa:
        raise Exception('could not find taxa: ' + ' '.join(missing_taxa))

    # write selected lines from each paragraph
    for p in paragraphs:
        for name in selected_taxa:
            i = name_to_index[name]
            print >> fout, p[i]
        print >> fout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', default='-',
            help='input interleaved phylip file (default stdin)')
    parser.add_argument('-o', default='-',
            help='output interleaved phylip file (default stdout)')
    parser.add_argument('taxa', nargs='+',
            help='taxa to include in the output')
    args = parser.parse_args()
    with fileutil.open_or_stdin(args.i) as fin:
        with fileutil.open_or_stdout(args.o, 'w') as fout:
            main(args.taxa, fin, fout)

