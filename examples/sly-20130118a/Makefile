
# Define the root directory that has the scripts and stuff in it.
# You might want to change this.
SCRIPTS = ../../bin

all: U0.mat U1.mat lam0.mat lam1.mat XQ.mat

U0.mat U1.mat lam0.mat lam1.mat XQ.mat: S0.mat S1.mat D0.mat D1.mat L.mat
	python $(SCRIPTS)/slowedml-sly.py \
		--debug \
		--S0-in S0.mat \
		--S1-in S1.mat \
		--D0-in D0.mat \
		--D1-in D1.mat \
		--L-in L.mat \
		--U0-out U0.mat \
		--U1-out U1.mat \
		--lam0-out lam0.mat \
		--lam1-out lam1.mat \
		--XQ-out XQ.mat

clean:
	rm -f U0.mat U1.mat lam0.mat lam1.mat XQ.mat
