# This is a demo inspired by the data and analyses in Yang-Nielsen 2008.
#
# This analysis uses only the first two genes.



# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SCRIPTS = ../../bin
GCODES = ../../genetic.codes



all: unique.pattern out.verts out.edges

out.verts out.edges: HCMMR.tree
	python $(SCRIPTS)/slowedml-expand-newick.py \
		--v-out out.verts --e-out out.edges < HCMMR.tree

full.phylip: HCMMR.txt.small
	python $(SCRIPTS)/slowedml-reformat-yn-2008.py \
		< HCMMR.txt.small > full.phylip

unique.weights unique.phylip: full.phylip
	python $(SCRIPTS)/slowedml-unique.py \
		-w unique.weights < full.phylip > unique.phylip

unique.pattern: unique.phylip
	python $(SCRIPTS)/slowedml-phylip-to-pattern.py \
		--code $(GCODES)/human.nuclear \
		< unique.phylip > unique.pattern

clean:
	rm -f *.phylip *.weights *.pattern *.verts *.edges
