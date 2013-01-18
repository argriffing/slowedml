# This is a demo inspired by the data and analyses in Yang-Nielsen 2008.
#
# This analysis uses only the first two genes.

all: unique.pattern

full.phylip: HCMMR.txt.small
	python ../../reformat-yang-2008.py < HCMMR.txt.small > full.phylip

unique.weights unique.phylip: full.phylip
	python ../../unique.py -w unique.weights < full.phylip > unique.phylip

unique.pattern: unique.phylip
	python ../../phylip-to-pattern.py \
		--code=../../genetic.codes/human.nuclear \
		< unique.phylip > unique.pattern

clean:
	rm *.phylip *.weights *.pattern

