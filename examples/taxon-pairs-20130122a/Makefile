# Compute substitution counts from an initial taxon to a final taxon.


# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SCRIPTS = ../../bin


all: computed.likelihood

computed.likelihood: example.transition.matrix example.distn computed.counts
	python $(SCRIPTS)/slowedml-likelihood-v1.py \
		--transition-matrix example.transition.matrix \
		--initial-distn example.distn \
		--counts computed.counts \
		-o computed.likelihood

computed.counts: example.taxa example.patterns example.weights
	python $(SCRIPTS)/slowedml-count-v1.py \
		--taxa example.taxa \
		--patterns example.patterns \
		--pattern-weights example.weights \
		--initial-taxon mouse \
		--final-taxon rat \
		--nstates 4 \
		--counts-out computed.counts

clean:
	rm -f computed.counts computed.likelihood

