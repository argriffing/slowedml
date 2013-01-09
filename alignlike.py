"""
Likelihoods involving alignment column patterns and multiplicities.

Each vertex is assumed to be an integer in a range starting with zero.
The patterns matrix has a row for each pattern and a column for each vertex.
Vertices that do not correspond to a sequence in the alignment will
have value -1 in the pattern matrix.
Other vertices will have a value that correspond to a state,
for example a nucleotide or a codon or whatever.
"""

def brute(
        ov, v_to_children, de_to_P, root_prior,
        patterns, pat_mults,
        ):
    """
    This function is only for testing and documentation.
    The P matrices and the root prior may be algopy objects.
    @param ov: ordered vertices with child vertices before parent vertices
    @param v_to_children: map from a vertex to a sequence of child vertices
    @param de_to_P: map from a directed edge to a transition matrix
    @param root_prior: equilibrium distribution at the root
    @param patterns: each pattern assigns a state to each leaf
    @param pat_mults: a multiplicity for each pattern
    """
