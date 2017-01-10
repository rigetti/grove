========
Overview
========
pyQAOA is a python module for running the Quantum Approximate Optimization
Algorithm on an instance of a quantum abstract machine.

Structure
_________

The pyQAOA package contains separate modules for each type of problem
instance: MAX-CUT, graph partitioning, etc.
For each problem instance the user specifics the driver Hamiltonian,
cost Hamiltonian, and the approximation order of the algorithm.

The package is structured as follows:

``qaoa.py`` contains the base QAOA class and routines for finding optimal
rotation angles via `Grove's quantum-variational-eigensolver method <../vqe/vqe.html>`_.

The following cost functions come standard with this package:

* ``maxcut_qaoa.py`` implements the cost function for MAX-CUT problems.

* ``numpartition_qaoa.py`` implements the cost function for bipartitioning a list of numbers.

* ``graphpartition_qaoa.py`` implements the cost function for graph partitioning--i.e. minimial cut between two subgraphs of equal size

* ``graphpartitioning_jaynescummingsdriver.py`` implements the graph partitioning problem with the constraint that the magnetization equals some constant `m`.
