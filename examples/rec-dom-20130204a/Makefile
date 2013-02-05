# Look at max log likelihoods for various D parameters.
#
# The D parameter follows Kimura's 1957 notation,
# and for our purposes it is taken to be defined
# with respect to the dominance or recessiveness of
# a mutant allele with greater fitness than the background allele.
# We have a codon substitution model that theoretically allows us to infer
# codon preference recessivity or dominance,
# and we will search for max log likelihoods when D is constrained
# to various values.
# We will plot these max log likelihoods as a function of D.
# We will use a subset of the data that Ziheng Yang posted to his website
# to accompany the 2008 publication by Yang and Nielsen.
# Because we are not so talented at software development,
# we will restrict our analysis to a single pair of taxa.
# We will also write in uncomfortably stilted prose.

# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SCRIPTS = ../../bin
GCODES = ../../genetic.codes
MUTSELDATA = ~/data/YangNielsen2008MBE.MutSel

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

# the trees file is like
# ((hg18, panTro2), rheMac2, (mm8, rn4));

INITIAL_TAXON = mm8
FINAL_TAXON = rn4

#INITIAL_TAXON = hg18
#FINAL_TAXON = panTro2

#INITIAL_TAXON = hg18
#FINAL_TAXON = rheMac2


all: my.table

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

my.table: computed.counts
	python $(SCRIPTS)/slowedml-rec-dom-20130204a.py \
		--code $(GCODES)/human.nuclear \
		--count-matrix computed.counts \
		--minimization-method $(MINIMIZATION) \
		--linspace-start -2 \
		--linspace-stop 2 \
		--linspace-num 5 \
		--table-out my.table \
		> debug.txt

clean:
	rm -f *.phylip *.weights *.pattern *.verts *.edges
	rm -f unique.*
	rm -f neg.log.likelihood ml.estimates ml.errors

