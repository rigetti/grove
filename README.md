Grove
----
A collection of quantum algorithms built using Forest.

Installation
------------

Grove depends on pyQuil, numpy, scipy, networkx, pytest, mock, and matplotlib.
To ensure binaries of these modules are installed along with the grove library
we recommend installing with `pip`.  First clone the Grove repo and `cd` into
it.

```
git clone https://github.com/rigetticomputing/grove.git
cd grove
```

Now the library can be installed with pip.

```
pip install -e ./
```

In order to access the Rigetti Forest service an API key or configuration file
will need to be installed.  Please see the
[pyQuil](https://github.com/rigetticomputing/pyQuil-dev) documentation.

Dependencies
------------

* Numpy
* Scipy
* pyQuil
* Mock (for development testing)
* NetworkX (for building and analyzing graphs)
* Matplotlib (useful for plotting)

Building the Docs
------------
To build the documentation run

```
sphinx-apidoc -o ./docs/source ./pyquil/
cd docs/
make html
```

To view the docs navigate to the `docs/_build` directory in the Grove root
directory and open the index.html file a browser. 

