Grove
=====

A collection of quantum algorithms built using the Rigetti Forest platform.
Grove is licensed under the [Apache 2.0 license](https://github.com/rigetticomputing/grove/blob/master/LICENSE).

[![Build Status](https://semaphoreci.com/api/v1/rigetti/grove/branches/master/badge.svg)](https://semaphoreci.com/rigetti/grove)
[![Documentation Status](https://readthedocs.org/projects/grove-docs/badge/?version=latest)](http://grove-docs.readthedocs.io/en/latest/?badge=latest)

Grove currently includes:

* [Quantum Teleportation](http://grove-docs.readthedocs.io/en/latest/teleportation.html)
* [The Variational-Quantum-Eigensolver (VQE)](http://grove-docs.readthedocs.io/en/latest/vqe.html)
* [The Quantum Approximate Optimization Algorithm (QAOA)](http://grove-docs.readthedocs.io/en/latest/qaoa.html)
* [The Quantum Fourier Transform (QFT)](http://grove-docs.readthedocs.io/en/latest/qft.html)
* [Phase Estimation Algorithm](http://grove-docs.readthedocs.io/en/latest/phaseestimation.html)

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

Dependencies
------------

* NumPy
* SciPy
* NetworkX (for building and analyzing graphs)
* Matplotlib (useful for plotting)
* pytest (optional, for development testing)
* mock (optional, for development testing)

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

We use pytest for testing. Tests can be run from the top-level directory using:
```
py.test
```


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
