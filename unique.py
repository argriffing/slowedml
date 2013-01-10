"""
Read an interleaved phylip file and extract the unique columns and their counts.

Note that the column uniqueness is by codon column not by nucleotide column.
The order of the codon columns should be stable.
"""

from collections import defaultdict
import argparse
import sys

def main(fin, fout, fout_weights):
    """
    @param fin: open file for reading interleaved phylip alignment
    @param fout: open file for writing interleaved phylip alignment
    @param fout_weights: open file for writing codon column multiplicities
    """

    # read rows of string sequences
    rows = []
    for line in fin:
        row = line.split()
        if row:
            rows.append(row)

    # init the list of unique columns
    unique_col_list = []
    col_to_count = defaultdict(int)

    # get the number of taxa and the total number of nucleotides
    s_ntaxa, s_nnucs = rows[0]
    ntaxa = int(s_ntaxa)
    nnucs = int(s_nnucs)

    # read the taxon names from the first paragraph
    taxon_names = []
    for row in rows[1:1+ntaxa]:
        taxon_names.append(row[0])

    # check that the data is in the expected format
    if len(rows) % ntaxa == 1:
        nparagraphs = (len(rows) - 1) / ntaxa
    else:
        raise Exception

    # go through the input rows, paragraph by paragraph
    for i in range(nparagraphs):

        # the first paragraph has taxon names prefixed to its rows
        paragraph = rows[i*ntaxa + 1 : (i+1)*ntaxa + 1]
        if i == 0:
            paragraph = [row[1:] for row in paragraph]

        # convert the paragraph into codon columns and look for unique cols
        codon_columns = zip(*paragraph)
        for col in codon_columns:
            if col not in col_to_count:
                unique_col_list.append(col)
            col_to_count[col] += 1

    # get some output formatting info
    name_lengths = [len(name) for name in taxon_names]
    ljust_spacing = max(name_lengths + [9])

    # write the interleaved phylip header
    nunique_codon_cols = len(unique_col_list)
    print >> fout, ' %d %d' % (ntaxa, 3 * nunique_codon_cols)

    # write the output files
    ncols_per_paragraph = 15
    offset = 0
    while True:

        # transpose the column list back into a paragraph
        cols = unique_col_list[offset : offset+ncols_per_paragraph]
        if not cols:
            break
        paragraph = zip(*cols)

        # write the weights corresponding to these columns
        if fout_weights is not None:
            weights = [col_to_count[col] for col in cols]
            print >> fout_weights, '\n'.join(str(w) for w in weights)
        
        # write the paragraph
        for i in range(ntaxa):
            row = paragraph[i]
            if offset:
                print >> fout, ''.ljust(ljust_spacing),
            else:
                print >> fout, taxon_names[i].ljust(ljust_spacing),
            print >> fout, ' '.join(row)
        print >> fout
        
        # move the the next paragraph worth of columns
        offset += ncols_per_paragraph


if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', default='-',
            help='read the interleaved phylip file from here')
    parser.add_argument('-o', default='-',
            help='write the phylip file with unique columns here')
    parser.add_argument('-w',
            help='write the column multiplicities here')

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
    if args.w:
        fout_weights = open(args.w, 'w')
    else:
        fout_weights = None

    # read and write the data
    main(fin, fout, fout_weights)

    # close the files
    if fin is not sys.stdin:
        fin.close()
    if fout is not sys.stdout:
        fout.close()
    if fout_weights is not None:
        fout_weights.close()

