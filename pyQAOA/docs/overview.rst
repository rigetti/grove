========
Overview
========
pyQAOA is a python module for running the Quantum Approximate Optimization
Algorithm on an instance of a quantum abstract machine.

Structure
_________

The pyQAOA package contains separate modules for each type of problem
instance: MAX-CUT, K-SAT, etc.
Each problem instance is a class that inherits the main QAOA object and
overrides the problem specific pieces of the algorithm: implementation of the
cost function and reference hamiltonian, generating the program that evolves the 
reference state to the problem's ground state, and bounds for the angles.

The package currently includes `maxcut_qaoa.py` which implements the cost
clauses and the driver hamiltonian for the MAX-CUT cost function.  

The package is structured as follows:

`qaoa.py` contains the base QAOA class and routines for finding the optimal
rotation angles via the quantum-variational-eigensolver method.

The following cost functions come with the package:
* `maxcut_qaoa.py` implements the cost function for MAX-CUT
problems.
* `numpartition_qaoa.py` implements the cost function for
bipartitioning a list of numbers.
* `graphparition_qaoa.py` implements the cost function for graph
  partitioning--i.e. minimial cut between two subgraphs of equal size
* `graphparitioning_jaynescummingsdriver.py` implements the graph paritioning
  problem with the constraint that the magnetization equals some constant `m`.
