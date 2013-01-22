slowedml
========

A slow implementation of some common functions
associated with a weird codon rate matrix.


requirements
------------

 * [Python](http://python.org/) 2.7+ (but not 3.x)
 * [NumPy](http://www.numpy.org/)
 * [SciPy](http://www.scipy.org/)
 * [pyfelscore](https://github.com/argriffing/pyfelscore)


optional python packages
------------------------

This is not currently used but may be used in the future.
 * [h5py](http://www.h5py.org/)


standard installation
---------------------

You can install this package
using the standard distutils installation procedure
for python packages with setup.py scripts,
as explained [here](http://docs.python.org/2/install/index.html).


install using pip
-----------------

One of several Python package installation helpers is called
[pip](http://www.pip-installer.org/).
You can use this to install directly from github using the command

`$ pip install --user https://github.com/argriffing/slowedml/zipball/master`

which can be reverted by

`$ pip uninstall slowedml`


testing the installation
------------------------

To test the installation of the python package, try running the command

`$ python -c "import slowedml; slowedml.test()"`

on your command line,
where the `$` is my notation for a shell prompt rather than
something that you are supposed to type.
Your command line prompt might look different.

