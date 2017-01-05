===========================
Installation and Quickstart
===========================

Dependencies
------------

Grove depends on a few scientific python packages as well as the python library for Quil:

* numpy
* scipy
* `pyQuil <https://github.com/rigetticomputing/pyQuil.git>`_
* NetworkX
* Matplotlib

Optional

* pytest (for testing)
* mock

**Numpy** and **Scipy** can be installed with `pip` (a python package manager). ::

    pip install numpy
    pip install scipy

Or using the Conda package manager ::

    conda install numpy
    conda install scipy

**pyQuil** can be installed by changing directory to where you would like to keep
the pyQuil repository and running ::

    git clone https://github.com/rigetticomputing/pyQuil.git
    cd pyQuil
    pip install -e .


You will need to make sure that your pyQuil installation is properly configured to run with a
QVM or quantum processor. See the pyQuil documentation for instructions on how to do this.


Grove Installation
-------------------

Clone the `git repository <https://github.com/rigetticomputing/grove.git>`_, `cd` into it, and
run ::

    pip install -e .

