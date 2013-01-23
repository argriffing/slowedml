# This is a demo inspired by the data and analyses in Yang-Nielsen 2008.
#
# This analysis uses the mitochondrial dna.


# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SCRIPTS = ../../bin
GCODES = ../../genetic.codes


all: log.likelihood

log.likelihood: computed.counts
	python $(SCRIPTS)/slowedml-yn-2008-ml.py \
		--code $(GCODES)/human.mitochondrial \
		--count-matrix computed.counts \
		-o log.likelihood \
		> debug.txt


computed.counts:
	python $(SCRIPTS)/slowedml-yn-2008-mito.py \
		-i mtCDNA.HC.txt \
		--code $(GCODES)/human.mitochondrial \
		--counts-out computed.counts

clean:
	rm -f computed.counts log.likelihood
