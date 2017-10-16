Grover's Search Algorithm and Amplitude Amplification
=====================================================

Overview
--------

This module implements Grover's Search Algorithm, and the more general Amplitude Amplification
Algorithm. Grover's Algorithm solves the following problem:

Given a collection of basis states {:math:`\ket{y}_i`}, and a quantum circuit that assigns these
states a common phase unique from the other basis states, construct a quantum circuit that when
given the uniform superposition :math:`\ket{s} = \frac{1}{\sqrt{N}}\sum\limits^{N-1}_{i=0}\ket{x_i}`
as input, will produce a state :math:`\ket{s'}` with nearly all of its support on the states
:math:`\{\ket{y}_i\}`.

Algorithm and Details
---------------------

Grover's Algorithm requires an oracle :math:`U_w`. An oracle is a quantum circuit with the following
structure: 

.. math::
  U_w: \ket{x}\ket{q} \to \ket{f(x)}\ket{q\oplus f(x)}

where :math:`f:\{0,1\}^n\to\{0,1\}^n`, and :math:`\ket{q}` is a single ancilla qubit. We see that if
we prepare the ancilla qubit :math:`\ket{q}` in the state :math:`\ket{-} =
\frac{1}{\sqrt{2}}(\ket{0} - \ket{1})` then :math:`U_w` takes on a particularly useful action on our
qubits:

.. math::
   U_w: \ket{x}\ket{-}\to\frac{1}{\sqrt{2}}\ket{x}(\ket{0\oplus f(x)} - \ket{1\oplus f(x)})
 
If :math:`f(x)=0`, then the ancilla qubit is left unchanged, however if :math:`f(x)=1` we see that
the ancilla picks up a phase factor of :math:`-1`. Thus, when used in conjunction with the ancilla
qubit, we may write the action of the oracle circuit on the data qubits :math:`\ket{q}` as:

.. math::
   U_w: \ket{q}\to(-1)^{f(x)}\ket{q}
      
The other gate of note in Grover's Algorithm is the so-called Diffusion operator. This operator is 
defined as:

.. math:: \mathcal{D} :=
   \begin{bmatrix}
   \frac{2}{N} - 1 & \frac{2}{N} & \dots &  \frac{2}{N} \\
   \frac{2}{N}\\
   \vdots          &             & \ddots \\ 
   \frac{2}{N}     &             &       &  \frac{2}{N} - 1
   \end{bmatrix}

Given these definitions we can now describe Grover's Algorithm. Given :math:`n + 1` qubits we
initialize them to the state :math:`\ket{s}\ket{-}`. Applying the oracle :math:`U_w` to the qubits
yields the state mentioned above, :math:`\sum\limits^{N-1}_{0}(-1)^{f(x)}\ket{x}\ket{-}`. Then we
apply the n-fold Hadamard gate :math:`H^{\otimes n}` to :math:`\ket{q}`, followed by
:math:`\mathcal{D}`, and then :math:`H^{\otimes n}` once more to :math:`\ket{q}`. It can be shown
[1]_ that if this process is iterated for :math:`\mathcal{O}(\sqrt{N})` iterations, a measurement of
:math:`\ket{q}` will result in one of :math:`\{y_i\}` with probability near one.

Source Code Docs
----------------

Here you can find documentation for the different submodules in amplification.

grove.amplification.amplification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: grove.amplification.amplification
    :members:
    :undoc-members:
    :show-inheritance:

grove.amplification.grover
~~~~~~~~~~~~~~~~~~~~~~~~~~

    .. automodule:: grove.amplification.grover
        :members:
        :undoc-members:
        :show-inheritance:

.. [1] https://en.wikipedia.org/wiki/Grover's_algorithm#Algebraic_proof_of_correctness
