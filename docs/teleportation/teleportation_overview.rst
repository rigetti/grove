=====================================
Introduction to Quantum Teleportation
=====================================
Quantum teleportation is a method for transmitting the information of a qubit
from one location to another.  The process relies on a previously shared
entangled state between the two locations and classical communication.

In the canonical description of quantum teleportation [1]_ [2]_ two parties (Alice
and Bob) are trying to transmit a state from one to another.  They start with
an Alice having half of an entangled bell pair (Bob having the other half) and
Alice with a third qubit that she would like to transmit to Bob.  Alice then
entangles the third qubit with her Bell pair qubit, measures both her qubits, and sends the
measurement result :math:`\{0, 1\}^{2}` to Bob.  Bob uses the classical information
from Alice to fix up his state and now has his qubit in the state of Alice's
original qubit she wanted to transfer.

Given that Alice's data qubit is labeled 0 and the Bell pair are labeled 1, 2
(Bob having qubit labeled 2) a Quil program performing the transfer is as
follows:

.. code::

    CNOT 0 1
    H 0
    MEASURE 0 [0]
    MEASURE 1 [1]
    JUMP-UNLESS @ NOX [1]
    X 2
    LABEL @NOX
    JUMP-UNLESS @NOZ [0]
    Z 2
    LABEL @NOZ

The Bell state preparation can be prepended to the above Quil

.. code::

    H 1
    CNOT 1 2

=======================
Teleportation in pyQuil
=======================
We have included pyQuil code that programatically generates a teleportation
program between any two qubits using one ancilla qubit.  The ancilla qubit is
the piece of the Bell pair that Alice holds.  Examples can be found in
`grove.teleport.teleportation.py`.

.. rubric:: References

.. [1] Nielsen, Michael A., and Isaac L. Chuang. Quantum computation and quantum information. Cambridge university press, 2010.

.. [2] Wikipedia contributors. "Quantum teleportation." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 4 Jan. 2017. Web. 4 Jan. 2017.


