"""
Convert an interleaved Phylip codon alignment into an array of integers.

The input genetic code file should be like one of the files
in the genetic.codes directory.
The input file for specify the taxon subset and ordering
should be a two-column tab-separated file,
where the first column is just integers starting with zero,
and the second column has the taxon names.
"""

import csv

import numpy as np

import phylip
import design


def main(fin, fin_gcode, fin_taxa, fout):
    """
    @param fin: interleaved phylip codon alignment file open for reading
    @param fin_gcode: open file for reading the genetic code
    @param fin_taxa: optional open file for defining taxon subset and order
    @param fout: open file for writing the integer ndarray as text
    """

    # read the description of the genetic code
    arr = list(csv.reader(fin_gcode, delimiter='\t'))
    indices, aminos, codons = zip(*arr)
    if [int(x) for x in indices] != range(len(indices)):
        raise ValueError

    # read the interleaved phylip alignment
    taxon_names = None
    cols = []
    for col in phylip.read_interleaved_codon_alignment(fin):
        if taxon_names is None:
            taxon_names = col
        else:
            cols.append(col)

    # try a user-defined ordered taxon subset
    if fin_taxa is not None:

        # read the ordered taxon subset
        arr = list(csv.reader(fin_taxa, delimiter='\t'))
        indices, requested_taxa = zip(*arr)
        if [int(x) for x in indices] != range(len(indices)):
            raise ValueError

        # construct the inverse map of the default taxon ordering
        name_to_i = dict((x, i) for i, x in enumerate(taxon_names))

        # Redefine the columns according to the user ordering and subsetting.
        # In this code we are pretending to be a database software.
        rows = zip(*cols)
        user_rows = []
        for name in enumerate(requested_taxa):
            user_rows.append(rows[name_to_i[name]])
        user_cols = zip(*user_rows)
        cols = user_cols

    # define the ndarray of integers
    M = design.get_pattern_array(codons, cols)

    # write the ndarray of integers
    np.savetxt(fout, M)


if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', required=True,
            help='an input file that defines the genetic code')
    parser.add_argument('--taxa',
            help='an input file that defines an ordered taxon subset')
    parser.add_argument('-i', default='-',
            help='read the interleaved phylip file from here')
    parser.add_argument('-o', default='-',
            help='write the corresponding integer array here')

    # read the args
    args = parser.parse_args()

    # open files for reading and writing
    if args.i == '-':
        fin = sys.stdin
    else:
        fin = open(args.i)
    if args.o == '-':
        fout = sys.stdout
    else:
        fout = open(args.o, 'w')
    if args.taxa:
        fin_taxa = open(args.taxa)
    else:
        fin_taxa = None

    # read and write the data
    with open(args.code, 'r') as fin_gcode:
        main(fin, fin_gcode, fin_taxa, fout)

    # close the files
    if fin_taxa is not None:
        fin_taxa.close()
    if fin is not sys.stdin:
        fin.close()
    if fout is not sys.stdout:
        fout.close()

