Phase Estimation Algorithm
==========================

Overview
--------

The phase estimation algorithm is a quantum subroutine useful for finding the
eigenvalue corresponding to an eigenvector \\(u\\) of some unitary operator.
It is the starting point for many other algorithms and relies on the inverse
quantum Fourier transform.  More details can be found in references [1]_.

Example
-------

First, connect to the QVM.

.. code-block:: python

    import pyquil.api as api

    qvm = api.QVMConnection()
    
Now we encode a phase into the unitary operator ``U``.

.. code-block:: python

    import numpy as np

    phase = 0.75
    phase_factor = np.exp(1.0j * 2 * np.pi * phase)
    U = np.array([[phase_factor, 0], 
                  [0, -1*phase_factor]])

Then, we feed this operator into the ``phase_estimation`` module. Here, we ask for 4
bits of precision.

.. code-block:: python

    from grove.alpha.phaseestimation.phase_estimation import phase_estimation

    precision = 4
    p = phase_estimation(U, precision)

Now, we run the program and check our output.

.. code-block:: python

    output = qvm.run(p, range(precision))
    wavefunction = qvm.wavefunction(p)

    print(output)
    print(wavefunction)

This should print the following:

.. code-block:: python

    [[0, 0, 1, 1]]
    (1+0j)|01100>

Note that .75, written as a binary fraction of precision 4, is 0.1100. Thus, we have
recovered the phase encoded into our unitary operator.

Source Code Docs
----------------

Here you can find documentation for the different submodules in phaseestimation.

grove.phaseestimation.phase_estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: grove.alpha.phaseestimation.phase_estimation
    :members:
    :undoc-members:
    :show-inheritance:

.. rubric:: References

.. [1] Nielsen, Michael A., and Isaac L. Chuang. Quantum Computation and Quantum Information. Cambridge University Press, 2010.
