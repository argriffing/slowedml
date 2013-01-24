# This is a demo inspired by the data and analyses in Yang-Nielsen 2008.
#
# This analysis uses the mitochondrial dna.


# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SCRIPTS = ../../bin
GCODES = ../../genetic.codes
PAML = ~/packages/paml4.6/bin

# defin the model
MODEL = FMutSel-F
#MODEL = F1x4
#MODEL = F1x4MG


all: log.likelihood

codeml: mtCDNA.HC.txt codeml.HC.ctl 2s.trees
	$(PAML)/codeml codeml.HC.ctl

log.likelihood: computed.counts
	python $(SCRIPTS)/slowedml-yn-2008-ml.py \
		--code $(GCODES)/human.mitochondrial \
		--count-matrix computed.counts \
		--$(MODEL) \
		-o log.likelihood \
		> debug.txt


computed.counts:
	python $(SCRIPTS)/slowedml-yn-2008-mito.py \
		-i mtCDNA.HC.txt \
		--code $(GCODES)/human.mitochondrial \
		--counts-out computed.counts

clean:
	rm -f computed.counts log.likelihood
