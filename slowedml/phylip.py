"""
Read a tiny subset of phylip interleaved alignment files.
"""


def read_interleaved_codon_alignment(fin):
    """
    Yield columns of the alignment.
    This function is not as stream-oriented as its interface may suggest.
    In particular, it eats tons of memory for no good reason.
    @param fin: a stream open for reading
    """

    # read rows of string sequences
    rows = []
    for line in fin:
        row = line.split()
        if row:
            rows.append(row)

    # init the list of columns
    col_list = []

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

    # yield a column consisting of the taxon names
    yield tuple(taxon_names)

    # go through the input rows, paragraph by paragraph
    for i in range(nparagraphs):

        # the first paragraph has taxon names prefixed to its rows
        paragraph = rows[i*ntaxa + 1 : (i+1)*ntaxa + 1]
        if i == 0:
            paragraph = [row[1:] for row in paragraph]

        # convert the paragraph into codon columns
        for column in zip(*paragraph):
            yield column
