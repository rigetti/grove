Arbitrary State Generation
==========================

Overview
--------

This module is concerned with making a program that can generate an arbitrary
state. In particular, if one is given a nonzero complex vector \\(\\mathbf{a}\\in\\mathbb{C}^N\\)
with components \\(a_i\\), the goal is to produce a program that takes in the state
\\(\\vert 0\\rangle\\) and outputs the state

$$
\\vert \\Psi \\rangle = \\sum_{i=0}^{N-1}\\frac{a_i}{\\vert \\mathbf{a}\\vert} \\vert i\\rangle
$$

where \\(\\vert i\\rangle\\) is interpreted by taking \\(i\\) in its binary representation.

This problem is approached in two different ways in this module, and will be described
in the sections to follow.
The first is to directly construct a circuit using a sequence of CNOT, rotation, Hadamard, and phase gates,
that produces the desired state.
The second is to construct a unitary matrix that could be decomposed into different
circuits depending on which gates one would see fit.

More details on the first approach can be found in references [1]_ and [2]_.


Arbitrary State Generation via Specific Circuit
-----------------------------------------------
The method in this approach follows the algorithm described in [1]_.
The idea is to imagine beginning with the desired state \\(\\vert \\Psi \\rangle\\).
First, controlled RZ gates are used to unify the phases of the coefficients
of consecutive pairs of basis states. Next, controlled RY gates are used to unify
the magnitudes (or probabilities) of those pairs of basis states, and hence unify the coefficients altogether.
Next, a swap is performed so that in subsequent steps, multiple pairs
of consecutive states will have the same pair of coefficients. This process
can be repeated, with each successive step of rotations requiring fewer controls due to the interspersed swaps.
Finally, with all states having the same coefficient, the Hadamard gate can be
applied to all the qubits to select out the \\(\\vert 0 \\rangle\\) state.
Lastly, a combination of a PHASE gate and RZ gate can be applied to remove the global phase.
The reverse of this program, which can be found by applying all gates in reverse
and all rotations with negated angles, this provides the desired program
for arbitrary state generation.

One key part of this algorithm is that each rotation step is uniformly controlled.
This has a relatively efficient decomposition into CNOTs and uncontrolled rotations,
and is the subject of reference [2]_.


Arbitrary State Generation via Unitary Matrix
---------------------------------------------
The method in this approach is to create a unitary operator mapping the ground state of a set
of qubits to the desired outcome state. This requires constructing a unitary matrix whose leftmost
column is \\(\\vert \\Psi \\rangle\\). By replacing the left column of the identity matrix with
\\(\\vert \\Psi \\rangle\\) and then QR factorizing it, one can construct such a matrix.


Source Code Docs
----------------

Here you can find documentation for the different submodules in arbitrary_state.

grove.arbitrary_state.arbitrary_state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: grove.alpha.arbitrary_state.arbitrary_state
    :members:
    :undoc-members:
    :show-inheritance:

grove.arbitrary_state.unitary_operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: grove.alpha.arbitrary_state.unitary_operator
    :members:
    :undoc-members:
    :show-inheritance:

.. rubric:: References

.. [1] http://140.78.161.123/digital/2016_ismvl_logic_synthesis_quantum_state_generation.pdf
.. [2] https://arxiv.org/pdf/quant-ph/0407010.pdf
