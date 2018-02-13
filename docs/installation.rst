Installation and Getting Started
================================

Prerequisites
-------------

Before you can start writing using Grove, you will need Python 2.7
(version 2.7.10 or greater) and the Python package manager pip. We recommend
installing `Anaconda <https://www.continuum.io/downloads>`__ for an all-in-one
installation of Python 2.7. If you don't have pip, it can be installed with
``easy_install pip``.

Installation
------------

You can install Grove directly from the Python package manager pip using:

::

    pip install quantum-grove

To instead install the bleeding-edge version from source,
clone the `Grove GitHub repository <https://github.com/rigetticomputing/grove.git>`_,
``cd`` into it, and run:

::

    pip install -e .

This will install Grove's dependencies if you do not already have them. The dependencies are:

* NumPy
* SciPy
* NetworkX
* Matplotlib
* pytest *(optional, for testing)*
* mock *(optional, for testing)*

Forest and pyQuil
-----------------

Grove also requires the Python library for Quil, called
`pyQuil <http://pyquil.readthedocs.io/en/latest/index.html>`_.

After obtaining the library from the `pyQuil GitHub repository <https://github.com/rigetticomputing/pyquil>`_
or from a source distribution, navigate into its directory in a terminal and run:

::

    pip install -e .

You will need to make sure that your pyQuil installation is properly configured to run with a
QVM or quantum processor (QPU) hosted on the `Rigetti Forest <http://forest.rigetti.com>`_, which
requires an API key. See the pyQuil `docs <http://pyquil.readthedocs.io/en/latest/index.html>`_
for instructions on how to do this.
