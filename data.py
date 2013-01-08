"""
This is for testing.

See pamlDOC.pdf on the internet for control file settings.
"""

g_nuc_a = """\
   3 1
t1  A
t2  A
t3  C
"""

g_trees_a = "(t1:1.0, t2:2.0, t3:3.0);"

g_ctl_a = """\
seqfile = demo_a.nuc
outfile = demo_a.out
treefile = demo_a.trees
noisy = 3
verbose = 1
runmode = 0
model = 0
Mgene = 0
ndata = 1
clock = 0
TipDate = 0
getSE = 1
Small_Diff = 1e-6
method = 0

fix_blength = 2
* -1 : start from random points
*  0 : ignore branch lengths
*  1 : use branch lengths as initial values for MLE
*  2 : branch lengths are fixed and are not estimated
"""

def main():
    """
    Spam some paml files into the current directory.
    """
    with open('demo_a.nuc', 'w') as fout:
        print >> fout, g_nuc_a
    with open('demo_a.trees', 'w') as fout:
        print >> fout, g_trees_a
    with open('demo_a.ctl', 'w') as fout:
        print >> fout, g_ctl_a

if __name__ == '__main__':
    main()
