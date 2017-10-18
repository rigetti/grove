Grover's Search Algorithm and Amplitude Amplification
=====================================================

Overview
--------

This module implements Grover's Search Algorithm, and the more general Amplitude Amplification
Algorithm. Grover's Algorithm solves the following problem:

Given a collection of basis states {:math:`\ket{y}_i`}, and a quantum circuit :math:`U_w` that 
performs the following: 

.. math::
  U_w: \ket{x}\ket{q} \to \ket{x}\ket{q\oplus f(x)}

where :math:`f(x)=1` iff :math:`\ket{x}\in\{\ket{y}_i\}`, construct a quantum circuit that when
given the uniform superposition :math:`\ket{s} = \frac{1}{\sqrt{N}}\sum\limits^{N-1}_{i=0}\ket{x_i}`
as input, produces a state :math:`\ket{s'}` that, when measured, produces a state :math:`\{y_i\}`
with probability near one.

As an example, take :math:`U_w: \ket{x}\ket{q} \to \ket{x\cdot\vec{1}}\ket{q\oplus x\cdot\vec{1}}`. 
In this case, :math:`f(x)=1` iff :math:`x=1`, and so starting with the state :math:`\ket{s}` we hope
end up with a state :math:`\ket{\psi}` such that :math:`\braket{\psi}{\vec{1}}\approx1`.

Algorithm and Details
---------------------

Grover's Algorithm requires an oracle :math:`U_w`, that performs the mapping as described above,
with :math:`f:\{0,1\}^n\to\{0,1\}^n`, and :math:`\ket{q}` a single ancilla qubit. We see that if
we prepare the ancilla qubit :math:`\ket{q}` in the state :math:`\ket{-} =
\frac{1}{\sqrt{2}}(\ket{0} - \ket{1})` then :math:`U_w` takes on a particularly useful action on our
qubits:

.. math::
   U_w: \ket{x}\ket{-}\to\frac{1}{\sqrt{2}}\ket{x}(\ket{0\oplus f(x)} - \ket{1\oplus f(x)})
 
If :math:`f(x)=0`, then the ancilla qubit is left unchanged, however if :math:`f(x)=1` we see that
the ancilla picks up a phase factor of :math:`-1`. Thus, when used in conjunction with the ancilla
qubit, we may write the action of the oracle circuit on the data qubits :math:`\ket{x}` as:

.. math::
   U_w: \ket{x}\to(-1)^{f(x)}\ket{x}
      
The other gate of note in Grover's Algorithm is the Diffusion operator. This operator is 
defined as:

.. math:: \mathcal{D} :=
   \begin{bmatrix}
   \frac{2}{N} - 1 & \frac{2}{N} & \dots &  \frac{2}{N} \\
   \frac{2}{N}\\
   \vdots          &             & \ddots \\ 
   \frac{2}{N}     &             &       &  \frac{2}{N} - 1
   \end{bmatrix}

This operator takes on its name from its similarity to a discretized version of the diffusion
equation Namely, if we consider :math:`\frac{\partial\psi(t)}{\partial t} = \nabla\cdot\nabla\psi(t)`,
and discretize our system such that at each time step a system of :math:`N` particles can diffuse to
each of their :math:`N-1` neighbors, it can be shown that the equations takes on the form
:math:`\psi(t + \Delta t) = \mathcal{D}'\psi` where :math:`\mathcal{D}'` has a form similar to
:math:`\mathcal{D}`.
      
One property that :math:`\mathcal{D}` has is that it inverts the amplitudes of an input state about
their mean. Thus, one way of viewing Grover's Algorithm is as follows. First, we flip the amplitude
of the desired state(s) with :math:`U_w`, then invert the amplitudes about their mean, which will
result in the amplitude of the desired state being slightly larger than all the other
amplitudes. Iterating this process will eventually result in the desired state having a
significantly larger amplitude. As short example by analogy, consider the vector of all ones,
:math:`[1, 1, ..., 1]`. Suppose we want to apply a transformation that increases the value of the
second input, and supresses all other inputs. We can first flip the sign to yield :math:`[1, -1, 1,
..., 1]` Then, if there are a large number of entries we see that the mean will be rougly one. Thus
inverting the entries about the mean will yield, approximately, :math:`[-1, 3, -1, ..., -1]`. Thus
we see that this procedure, after one iteration, significantly increases the amplitude of the
desired index with respect to the other indices. See [2]_ for more.

Given these definitions we can now describe Grover's Algorithm: 

Input:
  :math:`n + 1` qubits

Algorithm:
  #. Initialize them to the state :math:`\ket{s}\ket{-}`.

  #. Apply the oracle :math:`U_w` to the qubits, yielding 
     :math:`\sum\limits^{N-1}_{0}(-1)^{f(x)}\ket{x}\ket{-}`, where :math:`N = 2^n`

  #. Apply the n-fold Hadamard gate :math:`H^{\otimes n}` to :math:`\ket{x}`

  #. Apply :math:`\mathcal{D}`

  #. Apply :math:`H^{\otimes n}` to :math:`\ket{x}`

  #. Measure :math:`\ket{x}`

It can be shown [1]_ that if this process is iterate for :math:`\mathcal{O}(\sqrt{N})` iterations,
a measurement of :math:`\ket{x}` will result in one of :math:`\{y_i\}` with probability near one.

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

.. [1] Nielsen, M.A. and Chuang, I.L. Quantum computation and quantum information. Cambridge
       University Press, 2000. Chapter 6.
.. [2] https://arxiv.org/pdf/quant-ph/9605043.pdf
