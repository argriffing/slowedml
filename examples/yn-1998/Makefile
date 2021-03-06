# This demo is from Yang-Nielsen 1998 and it uses a phylip file from PAML.
#
# It uses max likelihood with a codon model and a tree with three branches.
#
# The codon alignment dNdSGene1.nuc originally had the following text:
# This is Acetylcholine receptor a, Gene # 1 in Ohta (1995).
# Table 2 of Yang & Nielsen (1998) used 456 codons (1368 nucleotides).
#

# Define the root directory that has the scripts and stuff in it.
SLOWEDML = ../..
PAML = ~/packages/paml4.6

SCRIPTS = $(SLOWEDML)/bin
GCODES = $(SLOWEDML)/genetic.codes


all: results.log

codeml: star.tree dNdSGene1.nuc my.codeml.ctl
	$(PAML)/bin/codeml my.codeml.ctl

star.tree:
	python $(SCRIPTS)/slowedml-create-star-tree.py -o star.tree \
		Rodents Primates Artiodactyla

dNdSGene1.verts dNdSGene1.edges: star.tree
	python $(SCRIPTS)/slowedml-expand-newick.py \
		-i star.tree \
		--v-out dNdSGene1.verts \
		--e-out dNdSGene1.edges

unique.weights unique.nuc: dNdSGene1.nuc
	python $(SCRIPTS)/slowedml-unique.py \
		-w unique.weights -i dNdSGene1.nuc -o unique.nuc

unique.pattern: dNdSGene1.verts unique.nuc
	python $(SCRIPTS)/slowedml-phylip-to-pattern.py \
		--code $(GCODES)/human.nuclear \
		--taxa dNdSGene1.verts \
		< unique.nuc > unique.pattern

results.log: unique.pattern unique.weights dNdSGene1.edges
	python $(SCRIPTS)/slowedml-yn-1998-ml.py \
		--code-in $(GCODES)/human.nuclear \
		--edges-in dNdSGene1.edges \
		--patterns-in unique.pattern \
		--weights-in unique.weights \
		> results.log

clean:
	rm -f unique.*
	rm -f star.tree
	rm -f *.verts *.edges
