=====================================
Variational-Quantum-Eigensolver (VQE)
=====================================

The Variational-Quantum-Eigensolver (VQE) [`1 <https://arxiv.org/abs/1304.3061>`_,
`2 <https://arxiv.org/abs/1509.04279>`_] is a quantum/classical hybrid
algorithm that can be used to find eigenvalues of a (often large) matrix :math:`H`.
When this algorithm is used in quantum simulations, :math:`H` is typically the Hamiltonian
of some system [`3 <https://arxiv.org/abs/1512.06860>`_, `4 <https://arxiv.org/abs/1602.01857>`_,
`5 <https://arxiv.org/abs/1510.03859>`_].
In this hybrid algorithm a quantum subroutine is run inside of a classical optimization loop.

The quantum subroutine has two fundamental steps:

1. Prepare the quantum state :math:`|\Psi(\vec{\theta})\rangle`, often called
   the *ansatz*.

2. Measure the expectation value :math:`\langle\,\Psi(\vec{\theta})\,|\,H\,|\,\Psi(\vec{\theta})\,\rangle`.

The `variational principle <https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics)>`_ ensures
that this expectation value is always greater than the smallest eigenvalue of :math:`H`.

This bound, allows us to use classical computation to run an optimization loop to find
this eigenvalue:

1. Use a classical non-linear optimizer to minimize the expectation value by varying
   ansatz parameters :math:`\vec{\theta}`.
2. Iterate until convergence.

Practically, the quantum subroutine of VQE amounts to preparing a state based off
of a set of parameters :math:`\vec{\theta}` and performing a series of measurements
in the appropriate basis. The paramaterized state (or ansatz) preperation can be tricky
in these algorithms and can dramatically effect performance.  Our vqe module allows any
Python function that returns a pyQuil program to be used as an ansatz generator.  This
function is passed into `vqe_run` as the `parameteric_state_evolve` argument. More details
are in the `source documentation <vqe_source.html#grove.pyvqe.vqe.VQE.vqe_run>`_.

Measurements are then performed on these states based on a Pauli operator decomposition of
:math:`H`. Using Quil, these measurements will end up in classical memory. Doing
this iteratively followed by a small amount of postprocessing, one may compute a real
expectation value for the classical optimizer to use.

In this documentation there is a very small first `example of VQE <vqe_example.html>`_ and the
implementation of the `Quantum Approximate Optimization Algorithm QAOA <../qaoa/qaoa.html>`_ also
makes use of the VQE module.

.. toctree::
   :name: mastertoc
   :maxdepth: 3

   vqe/vqe_example
   vqe/vqe_source
