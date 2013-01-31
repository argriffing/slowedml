# This should be a parametric bootstrap analysis.

# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SCRIPTS = ../../bin
GCODES = ../../genetic.codes
MUTSELDATA = ~/data/YangNielsen2008MBE.MutSel

# Define the number of simulations to run for each of the two models.
NSAMPLES = 100

# Model options:
#FMutSel-F    # mutation selection balance yang-nielsen 2008
#FMutSelPD-F  # fmutsel variant where preferred allele is dominant
#FMutSelPR-F  # fmutsel variant where preferred allele is recessive
#F1x4         # goldman-yang codon model
#F1x4MG       # muse-gaut codon model

# These are the two models that are to be compared to each other
# using "parametric bootstrap"
# where the first analysis treats the first model as the null model
# and the second analysis treats the second model as the null model.
FIRST_MODEL = FMutSel-F
SECOND_MODEL = FMutSelPD-F

# the trees file is like
# ((hg18, panTro2), rheMac2, (mm8, rn4));

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



all: sim.$(FIRST_MODEL).table sim.$(SECOND_MODEL).table


##############################################################################
# These calls extract the max likelihood estimates under the two models.

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

$(FIRST_MODEL).ml.estimates: computed.counts
	python $(SCRIPTS)/slowedml-yn-2008-ml.py \
		--code $(GCODES)/human.nuclear \
		--count-matrix computed.counts \
		--$(FIRST_MODEL) \
		--minimization-method $(MINIMIZATION) \
		--parameter-estimates-out $(FIRST_MODEL).ml.estimates

$(SECOND_MODEL).ml.estimates: computed.counts
	python $(SCRIPTS)/slowedml-yn-2008-ml.py \
		--code $(GCODES)/human.nuclear \
		--count-matrix computed.counts \
		--$(SECOND_MODEL) \
		--minimization-method $(MINIMIZATION) \
		--parameter-estimates-out $(SECOND_MODEL).ml.estimates


##############################################################################
# Set up and run the simulation.

computed.count.vector: computed.counts
	python $(SCRIPTS)/slowedml-yn-2008-loadtxt-count-vector.py \
		-i computed.counts \
		-o computed.count.vector

sim.$(FIRST_MODEL).table: computed.count.vector $(FIRST_MODEL).ml.estimates
	python $(SCRIPTS)/slowedml-yn-2008-sim.py \
		--empirical-codon-counts computed.count.vector \
		--simulation-parameter-values $(FIRST_MODEL).ml.estimates \
		--simulation-model $(FIRST_MODEL) \
		--first-inference-model $(FIRST_MODEL) \
		--second-inference-model $(SECOND_MODEL) \
		--code $(GCODES)/human.nuclear \
		--minimization-method $(MINIMIZATION) \
		--nsamples $(NSAMPLES) \
		--table-out sim.$(FIRST_MODEL).table

sim.$(SECOND_MODEL).table: computed.count.vector $(FIRST_MODEL).ml.estimates
	python $(SCRIPTS)/slowedml-yn-2008-sim.py \
		--empirical-codon-counts computed.count.vector \
		--simulation-parameter-values $(SECOND_MODEL).ml.estimates \
		--simulation-model $(SECOND_MODEL) \
		--first-inference-model $(SECOND_MODEL) \
		--second-inference-model $(FIRST_MODEL) \
		--code $(GCODES)/human.nuclear \
		--minimization-method $(MINIMIZATION) \
		--nsamples $(NSAMPLES) \
		--table-out sim.$(SECOND_MODEL).table


##############################################################################
# Delete the junk from the Makefile directory.

clean:
	rm -f *.phylip *.weights *.pattern *.verts *.edges
	rm -f two.taxon.tree two.taxon.phylip
	rm -f unique.*
	rm -f neg.log.likelihood ml.estimates ml.errors

