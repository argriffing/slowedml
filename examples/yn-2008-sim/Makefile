# Check whether I can distinguish between the FMutSel-F and FMutSelPD-F models.

# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SCRIPTS = ../../bin
GCODES = ../../genetic.codes
MUTSELDATA = ~/data/YangNielsen2008MBE.MutSel

# Model options:
#FMutSel-F    # mutation selection balance yang-nielsen 2008
#FMutSelPD-F  # fmutsel variant where preferred allele is dominant
#FMutSelPR-F  # fmutsel variant where preferred allele is recessive
#F1x4         # goldman-yang codon model
#F1x4MG       # muse-gaut codon model

# Define the model to estimate the parameters to be used for sampling.
# This model will give maximum likelihood estimates of six parameters:
# the expected number of substitutions, kappa, omega,
# and three nucleotide frequency ratios.
ROUGH_MODEL = FMutSel-F

# Sample according to one model and get the log likelihood ratio.
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


all: sim.table


##############################################################################
# These calls extract the max likelihood estimates under the rough model.

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
	python $(SCRIPTS)/slowedml-yn-2008-ml.py \
		--code $(GCODES)/human.nuclear \
		--count-matrix computed.counts \
		--$(ROUGH_MODEL) \
		--minimization-method $(MINIMIZATION) \
		--parameter-estimates-out ml.estimates \
		--parameter-errors-out ml.errors \
		--neg-log-likelihood-out neg.log.likelihood \
		> debug.txt


##############################################################################
# Set up and run the simulation.

computed.count.vector: computed.counts
	python $(SCRIPTS)/slowedml-yn-2008-loadtxt-count-vector.py \
		-i computed.counts \
		-o computed.count.vector

sim.$(FIRST_MODEL).table: computed.count.vector ml.estimates
	python $(SCRIPTS)/slowedml-yn-2008-sim.py \
		--empirical-codon-counts computed.count.vector \
		--simulation-parameter-values ml.estimates \
		--simulation-model $(FIRST_MODEL) \
		--first-inference-model $(FIRST_MODEL) \
		--second-inference-model $(SECOND_MODEL) \
		--code $(GCODES)/human.nuclear \
		--minimization-method $(MINIMIZATION) \
		--nsamples 1000 \
		--table-out sim.$(FIRST_MODEL).table

sim.$(SECOND_MODEL).table: computed.count.vector ml.estimates
	python $(SCRIPTS)/slowedml-yn-2008-sim.py \
		--empirical-codon-counts computed.count.vector \
		--simulation-parameter-values ml.estimates \
		--simulation-model $(SECOND_MODEL) \
		--first-inference-model $(SECOND_MODEL) \
		--second-inference-model $(FIRST_MODEL) \
		--code $(GCODES)/human.nuclear \
		--minimization-method $(MINIMIZATION) \
		--nsamples 1000 \
		--table-out sim.$(SECOND_MODEL).table


##############################################################################
# Delete the junk from the Makefile directory.

clean:
	rm -f *.phylip *.weights *.pattern *.verts *.edges
	rm -f two.taxon.tree two.taxon.phylip
	rm -f unique.*
	rm -f neg.log.likelihood ml.estimates ml.errors

