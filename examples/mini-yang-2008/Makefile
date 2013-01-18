# This is a demo inspired by the data and analyses in Yang-Nielsen 2008.
#
# This analysis uses only the first two genes.



# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SLOWEDML = ../..



all: unique.pattern out.verts out.edges

out.verts out.edges: HCMMR.tree
	python $(SLOWEDML)/expand-newick.py \
		--v-out out.verts --e-out out.edges < HCMMR.tree

full.phylip: HCMMR.txt.small
	python $(SLOWEDML)/reformat-yang-2008.py \
		< HCMMR.txt.small > full.phylip

unique.weights unique.phylip: full.phylip
	python $(SLOWEDML)/unique.py \
		-w unique.weights < full.phylip > unique.phylip

unique.pattern: unique.phylip
	python $(SLOWEDML)/phylip-to-pattern.py \
		--code $(SLOWEDML)/genetic.codes/human.nuclear \
		< unique.phylip > unique.pattern

clean:
	rm -f *.phylip *.weights *.pattern *.verts *.edges
