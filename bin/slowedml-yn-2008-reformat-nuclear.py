#!/usr/bin/env python

"""
Reformat the data in Ziheng Yang's file YangNielsen2008MBE.MutSel/HCMMR.txt .

This data file consist of a bunch of Phylip gene files concatenated together.
This script filters the file by converting it into a single
interleaved phylip file for which all of the genes for each
taxon have been concatenated.
The script uses stdin and stdout.
I am going for the following look.
http://emboss.sourceforge.net/docs/themes/seqformats/phylip
"""

import sys

def main(fin, fout):
    """
    @param fin: open input file
    @param fout: open output file
    """

    # read rows of string sequences
    rows = []
    for line in fin:
        row = line.split()
        if row:
            rows.append(row)

    # get the number of taxa and the length of the first gene
    s_ntaxa, s_ncols_first = rows[0]
    ntaxa = int(s_ntaxa)
    ncols_first = int(s_ncols_first)

    # read the taxon names from the first gene
    taxon_names = []
    for row in rows[1:1+ntaxa]:
        taxon_names.append(row[0])

    # check that the data is in the expected format
    nrows_per_gene = ntaxa + 1
    if len(rows) % nrows_per_gene == 0:
        ngenes = len(rows) / nrows_per_gene
    else:
        raise Exception

    # read the total number of columns of all genes
    ncols_total = 0
    for i in range(ngenes):
        row = rows[i * nrows_per_gene]
        if len(row) != 2:
            raise Exception
        s_ntaxa, s_ncols = row
        if int(s_ntaxa) != ntaxa:
            raise Exception('each gene should have the same number of taxa')
        ncols_gene = int(s_ncols)
        ncols_total += ncols_gene

    # concatenate the columns
    arr = [[] for i in range(ntaxa)]
    for gene_index in range(ngenes):
        for gene_row_index in range(ntaxa):
            row = rows[gene_index * nrows_per_gene + 1 + gene_row_index]
            taxon_name = row[0]
            taxon_index = taxon_names.index(taxon_name)
            arr[taxon_index].extend(row[1:])

    # write the interleaved phylip header
    print >> fout, ' %d %d I' % (ntaxa, ncols_total)

    # write the interleaved phylip data
    is_finished = False
    name_lengths = [len(name) for name in taxon_names]
    ljust_spacing = max(name_lengths + [8]) + 1
    chunks_per_paragraph = 15
    offset = 0
    while not is_finished:
        for i in range(ntaxa):
            chunks = arr[i][offset : offset + chunks_per_paragraph]
            if not chunks:
                is_finished = True
                break
            if offset:
                print >> fout, ''.ljust(ljust_spacing),
            else:
                print >> fout, taxon_names[i].ljust(ljust_spacing),
            print >> fout, ' '.join(chunks)
        if is_finished:
            break
        print >> fout
        offset += chunks_per_paragraph

if __name__ == '__main__':
    main(sys.stdin, sys.stdout)

