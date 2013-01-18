# This demo is from Yang-Nielsen 1998 and it uses a phylip file from PAML.
#
# It uses max likelihood with a codon model and a tree with three branches.
#
# The codon alignment dNdSGene1.nuc originally had the following text:
# This is Acetylcholine receptor a, Gene # 1 in Ohta (1995).
# Table 2 of Yang & Nielsen (1998) used 456 codons (1368 nucleotides).
#


# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SLOWEDML = ../..


all: unique.pattern

unique.weights unique.nuc: dNdSGene1.nuc
	python $(SLOWEDML)/unique.py \
		-w unique.weights -i dNdSGene1.nuc -o unique.nuc

unique.pattern: unique.nuc
	python $(SLOWEDML)/phylip-to-pattern.py \
		--code $(SLOWEDML)/genetic.codes/human.nuclear \
		--taxa dNdSGene1.verts \
		< unique.nuc > unique.pattern

clean:
	rm -f unique.*
