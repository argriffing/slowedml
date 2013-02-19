# Attempt to reproduce results from Yang and Nielsen (2008) on nuclear DNA.
#
# FIXME: change this file from omegamix to mgmix

# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SCRIPTS = ../../bin
GCODES = ../../genetic.codes
PAML = ~/packages/paml4.6
MUTSELDATA = ~/data/YangNielsen2008MBE.MutSel

# the trees file is like
# ((hg18, panTro2), rheMac2, (mm8, rn4));
#
# define the model
#MODEL = FMutSel-F-mix     # mutation selection balance yang-nielsen 2008
MODEL = FMutSelG-F-mix    # fmutsel variant with unconstrained recessivity

#MINIMIZATION = Nelder-Mead
#MINIMIZATION = Powell
#MINIMIZATION = CG
#MINIMIZATION = BFGS
#MINIMIZATION = Newton-CG
#MINIMIZATION = Anneal
MINIMIZATION = L-BFGS-B
#MINIMIZATION = TNC
#MINIMIZATION = COBYLA
#MINIMIZATION = SLSQP

#INITIAL_TAXON = mm8
#FINAL_TAXON = rn4

#INITIAL_TAXON = hg18
#FINAL_TAXON = panTro2

INITIAL_TAXON = hg18
FINAL_TAXON = rheMac2


all: neg.log.likelihood

codeml: two.taxon.phylip two.taxon.tree
	$(PAML)/bin/codeml my.codeml.ctl

two.taxon.tree:
	python $(SCRIPTS)/slowedml-create-star-tree.py \
		-o two.taxon.tree \
		$(INITIAL_TAXON) $(FINAL_TAXON)

two.taxon.phylip: full.phylip
	python $(SCRIPTS)/slowedml-yn-2008-extract-pair.py \
		-i full.phylip \
		-o two.taxon.phylip \
		$(INITIAL_TAXON) $(FINAL_TAXON)

out.verts out.edges:
	python $(SCRIPTS)/slowedml-expand-newick.py \
		--v-out out.verts \
		--e-out out.edges \
		-i $(MUTSELDATA)/HCMMR.trees \
		--nskiplines 2

full.phylip:
	python $(SCRIPTS)/slowedml-yn-2008-reformat-nuclear.py \
		< $(MUTSELDATA)/HCMMR.txt \
		> full.phylip

unique.weights unique.phylip: full.phylip
	python $(SCRIPTS)/slowedml-unique.py \
		-w unique.weights \
		< full.phylip \
		> unique.phylip

unique.pattern: unique.phylip
	python $(SCRIPTS)/slowedml-phylip-to-pattern.py \
		--code $(GCODES)/human.nuclear \
		< unique.phylip \
		> unique.pattern

computed.counts: out.verts unique.pattern unique.weights
	python $(SCRIPTS)/slowedml-count-v1.py \
		--taxa out.verts \
		--patterns unique.pattern \
		--pattern-weights unique.weights \
		--initial-taxon-name $(INITIAL_TAXON) \
		--final-taxon-name $(FINAL_TAXON) \
		--nstates 64 \
		--counts-out computed.counts

neg.log.likelihood ml.estimates ml.errors: computed.counts
	python $(SCRIPTS)/slowedml-yn-2008-ml-mgmix.py \
		--code $(GCODES)/human.nuclear \
		--count-matrix computed.counts \
		--minimization-method $(MINIMIZATION) \
		--$(MODEL) \
		> debug.txt

clean:
	rm -f *.phylip *.weights *.pattern *.verts *.edges
	rm -f two.taxon.tree two.taxon.phylip
	rm -f unique.*
	rm -f neg.log.likelihood ml.estimates ml.errors

