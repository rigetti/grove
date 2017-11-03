=======================
Deutsch-Jozsa Algorithm
=======================

Overview
--------

The Deutsch-Jozsa algorithm can determine whether a function mapping all bitstrings to a single bit
is constant or balanced, provided that it is one of the two. A constant function always maps to
either 1 or 0, and a balanced function maps to 1 for half of the inputs and maps to 0 for the other
half. Unlike any deterministic classical algorithm, the Deutsch-Jozsa Algorithm can solve this
problem with a single iteration, regardless of the input size. It was one of the first known quantum
algorithms that showed an exponential speedup, albeit against a deterministic (non-probabilistic)
classical compuetwr, and with access to a blackbox function that can evaluate inputs to the chosen
function.

Algorithm and Details
---------------------

This algorithm takes as input :math:`n` qubits in state :math:`\ket{x}`, an ancillary qubit in state
:math:`\ket{q}`, and additionally a quantum circuit :math:`U_w` that performs the following:

.. math::
   U_w: \ket{x}\ket{q}\to\ket{x}\ket{f(x)\oplus q}

In the case of the Deutsch-Jozsa algorithm, the function :math:`f` is some function mapping from
bitstrings to bits:

.. math::
   f: \{0,1\}^n\to\{0, 1\}

and is assumed to either be \textit{constant} or \textit{balanced}. Constant means that on all
inputs :math:`f` takes on the same value, and balanced means that on half of the inputs :math:`f`
takes on one value, and on the other half :math:`f` takes on a different value. (Here the value is
restricted to :math:`\{0, 1\}`)

We can then describe the algorithm as follows:

Input:
   :math:`n + 1` qubits
Algorithm:
  #. Prepare the \textit{ancilla} (:math:`\ket{q}` above) in the :math:`\ket{1}` state by performing
     an :math:`X` gate.
  #. Perform the :math:`n + 1`-fold Hadamard gate :math:`H^{\otimes n + 1}` on the :math:`n + 1`
     qubits.
  #. Apply the circuit :math:`U_w`.
  #. Apply the :math:`n`-fold Hadamard gate :math:`H^{\otimes n}` on the data qubits, :math:`\ket{x}`.
  #. Measure :math:`\ket{x}`. If the result is all zeroes, then the function is constant. Otherwise, it
     is balanced.

Implementation Notes
--------------------

The oracle in the :term:`Deutsch-Jozsa` module is not implemented in such a way that calling
`Deutsch_Jozsa.is_constant()` will yield an exponential speedup over classical implementations.  To
construct the quantum algorithm that is executing on the QPU we use a Quil `defgate`, which
specifies the circuit :math:`U_w` as its action on the data qubits :math:`\ket{x}`. This matrix is
exponentially large, and thus even generating the program will take exponential time.


Source Code Docs
----------------

Here you can find documentation for the different submodules in deutsch-jozsa.

grove.deutsch_jozsa.deutsch_jozsa.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: grove.deutsch_jozsa.deutsch_jozsa
    :members:
    :undoc-members:
    :show-inheritance:
