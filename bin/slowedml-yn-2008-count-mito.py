#!/usr/bin/env python

"""
Reformat the data in the file YangNielsen2008MBE.MutSel/mtCDNA.HC.txt.

This is from Ziheng Yang and Rasmus Nielsen's maximum likeilhood analyses.
"""

import argparse
import csv

import numpy as np

from slowedml import fileutil, design


def gen_paragraphs(fin):
    paragraph = []
    for line in fin:
        line = line.strip()
        if not line:
            if paragraph:
                yield paragraph
            paragraph = []
        else:
            paragraph.append(line)
    if paragraph:
        yield paragraph


def main(args):

    # read the description of the genetic code
    with open(args.code) as fin_gcode:
        arr = list(csv.reader(fin_gcode, delimiter='\t'))
        indices, aminos, codons = zip(*arr)
        if [int(x) for x in indices] != range(len(indices)):
            raise ValueError

    # read the input
    with fileutil.open_or_stdin(args.i) as fin:
        paragraphs = list(gen_paragraphs(fin))

    human_header = paragraphs[1][0]
    human_lines = paragraphs[1][1:]
    chimp_header = paragraphs[2][0]
    chimp_lines = paragraphs[2][1:]

    if human_header != 'Human_Horai':
        raise ValueError
    if chimp_header != 'Chimp_Horai':
        raise ValueError

    human_dna = ''.join(human_lines)
    human_codons = [human_dna[i:i+3] for i in range(0, len(human_dna), 3)]

    chimp_dna = ''.join(chimp_lines)
    chimp_codons = [chimp_dna[i:i+3] for i in range(0, len(chimp_dna), 3)]

    codon_alignment_columns = zip(*(human_codons, chimp_codons))

    patterns = design.get_pattern_array(codons, codon_alignment_columns)

    ncodons = len(codons)
    counts = np.zeros((ncodons, ncodons), dtype=int)
    for i, j in patterns:
        counts[i, j] += 1

    # write the (ncodons, ncodons) array of counts of human to chimp changes
    with fileutil.open_or_stdout(args.counts_out, 'w') as fout:
        np.savetxt(fout, counts, fmt='%g', delimiter='\t')


if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', default='-',
            help='the input data file (default is stdin)')
    parser.add_argument('--code', required=True,
            help='path to the human mitochondrial genetic code')
    parser.add_argument('--counts-out', default='-',
            help='write the matrix of codon changes here (default is stdout)')

    main(parser.parse_args())

