Grove
=====

A collection of quantum algorithms built using the Rigetti Forest platform.
Grove is licensed under the [Apache 2.0 license](https://github.com/rigetticomputing/grove/blob/master/LICENSE).

[![Build Status](https://semaphoreci.com/api/v1/rigetti/grove/branches/master/badge.svg)](https://semaphoreci.com/rigetti/grove)
[![Documentation Status](https://readthedocs.org/projects/grove-docs/badge/)](http://grove-docs.readthedocs.io/en/latest/)

Grove currently includes:

* [The Variational-Quantum-Eigensolver (VQE)](http://grove-docs.readthedocs.io/en/latest/vqe.html)
* [The Quantum Approximate Optimization Algorithm (QAOA)](http://grove-docs.readthedocs.io/en/latest/qaoa.html)
* [The Quantum Fourier Transform (QFT)](http://grove-docs.readthedocs.io/en/latest/qft.html)
* [Phase Estimation Algorithm](http://grove-docs.readthedocs.io/en/latest/phaseestimation.html)

Features in the `alpha` package are considered experimental.

Documentation
-------------

Documentation is hosted at [http://grove-docs.readthedocs.io/en/latest/](http://grove-docs.readthedocs.io/en/latest/)

Installation
------------

You can install Grove directly from the Python package manager `pip` using:
```
pip install quantum-grove
```

To instead install Grove from source, clone this repository, `cd` into it, and run:
```
pip install -e .
```

This will install Grove’s dependencies if you do not already have them.
However, you will still need to install pyQuil and set up a connection to
the Rigetti Forest (see below).
To enable the tomography module, you will also have to install qutip
and cvxpy, see below for more details.

Forest and pyQuil
-----------------

Grove also requires the Python library for Quil, called
[pyQuil](http://pyquil.readthedocs.io/en/latest/index.html).

You can install pyQuil directly from the Python package manager `pip` using:
```
pip install pyquil
```

To instead install pyQuil from source, clone the
[pyQuil GitHub repository](https://github.com/rigetticomputing/pyquil),
`cd` into it, and run:
```
pip install -e .
```

You will need to make sure that your pyQuil installation is properly
configured to run with a quantum virtual machine (QVM) or real quantum processor
(QPU) hosted on the  [Rigetti Forest](forest.rigetti.com), which requires an API key.
See the pyQuil [docs](http://pyquil.readthedocs.io/en/latest/index.html) for
instructions on how to do this.

Installing the dependencies for Quantum Tomography
--------------------------------------------------

Quantum tomography relies on the external packages qutip and and cvxpy,
which can be somewhat tricky to install.

You can first attempt to just run
```
pip install -r requirements.txt
pip install -r optional-requirements.txt
pip install quantum-grove
```

If the installation of the optional requirements fails, you can manually
install the individual packages as

```
pip install cython==0.24.1 scs==1.2.6
pip install qutip==4.1 cvxpy==0.4.11
```
These are not the most recent versions but they are the only ones that
have consistently worked for us across different platforms and python
versions.

For **Windows users**: Both qutip and cvxpy are fairly tricky to
install under windows and we therefore recommend using Anaconda's
``conda`` package manager to install these first and then ``pip``
to install ``quantum-grove``.


Building the Docs
-----------------

We use sphinx to build the documentation. To do this, navigate into Grove's top-level directory and run:

```
sphinx-build -b html docs/ docs/_build
```

To view the docs navigate to the newly-created `docs/_build` directory and open
the `index.html` file in a browser. Note that we use the Read the Docs theme for
our documentation, so this may need to be installed using `pip install sphinx_rtd_theme`.

Development and Testing
-----------------------

We use tox and pytest for testing. Tests can be executed from the top-level directory by simply
running:
```
tox
```
The setup is currently testing Python 2.7 and Python 3.6.


## How to cite Grove and Forest

If you use pyquil, grove or other parts of Forest in your research, please cite it as follows:

bibTeX:
```
@misc{1608.03355,
  title={A Practical Quantum Instruction Set Architecture},
  author={Smith, Robert S and Curtis, Michael J and Zeng, William J},
  journal={arXiv preprint arXiv:1608.03355},
  year={2016}
}
```

Text:
```
R. Smith, M. J. Curtis and W. J. Zeng, "A Practical Quantum Instruction Set Architecture," (2015), 
  arXiv:1608.03355 [quant-ph], https://arxiv.org/abs/1608.03355
```
