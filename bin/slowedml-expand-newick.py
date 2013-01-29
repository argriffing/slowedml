#!/usr/bin/env python

"""
Expand a newick tree representation into a vertex file and an edge file.

This requires DendroPy.
The vertex file will list vertices ordered so that
each child vertex has a smaller index than its parent vertex
according to the implicit rooting in the input newick file.
The edge file will list directed edges so that the
parent vertex precedes the child vertex according to the
rooting implicit in the newick file,
but the list of edges itself does not have a particular ordering.
The output files will have extremely simple formats.
If output files are not specified,
the output will be spammed to stdout.
"""

import argparse
import sys

import dendropy

from slowedml import moretypes


def get_short_node_desc(node):
    if node:
        return '%s : %s' % (id(node), node.taxon)
    else:
        return str(None)

def main(args, fin, fout_vertices, fout_edges):
    for line in range(args.nskiplines):
        line = fin.readline()
    tree = dendropy.Tree(
            stream=fin,
            schema='newick',
            )

    # make a postorder node list
    nodes = []
    nodes.extend(tree.leaf_iter())
    nodes.extend(tree.postorder_internal_node_iter())

    # invent taxon names
    for i, node in enumerate(nodes):
        if node.taxon is None:
            node.taxon = 'unknown_%d' % i

    # create a map from node id to node index
    node_id_to_index = dict((id(node), i) for i, node in enumerate(nodes))

    # make an edge list
    edges = []
    for edge in tree.preorder_edge_iter():
        tail = edge.tail_node
        head = edge.head_node
        if tail and head:
            #print get_short_node_desc(tail)
            #print get_short_node_desc(head)
            index_pair = (
                    node_id_to_index[id(tail)],
                    node_id_to_index[id(head)],
                    )
            edges.append(index_pair)

    # write the vertex file
    for i, node in enumerate(nodes):
        row = (i, node.taxon)
        print >> fout_vertices, '\t'.join(str(x) for x in row)

    # write the edge file
    for pair in edges:
        print >> fout_edges, '\t'.join(str(x) for x in pair)


if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i',
            help='read the newick tree from this file')
    parser.add_argument('--nskiplines',
            type=moretypes.nonneg_int, default=0,
            help='skip this many lines of the tree file')
    parser.add_argument('--v-out',
            help='write the ordered vertices to this file')
    parser.add_argument('--e-out',
            help='write the directed edges to this file')

    # get the args from the command line
    args = parser.parse_args()

    # open files for reading and writing
    fin = sys.stdin
    fout_vertices = None
    fout_edges = None
    if args.i:
        fin = open(args.i)
    if args.v_out:
        fout_vertices = open(args.v_out, 'w')
    if args.e_out:
        fout_edges = open(args.e_out, 'w')

    # read and write the data
    main(args, fin, fout_vertices, fout_edges)

    # close the files
    if fin is not sys.stdin:
        fin.close()
    if fout_vertices is not None:
        fout_vertices.close()
    if fout_edges is not None:
        fout_edges.close()

