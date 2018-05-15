Jordan Gradient
==========================

Overview
--------

This is an implementation of Stephen Jordan's "Fast Quantum Algorithm for
Numerical Gradient Estimation" [1]_. Jordan's algorithm utilizes an oracle to sample some
function at some distance, h, from the point of evaluation. 
These samples are then `kicked back` to output register via modular addition.
A quantum Fourier transform (QFT) is then used to transform the output register and
recover the oracle outputs. To first approximation, this can be used to
estimate the gradient of the function of interest.

In our implementation, the ``phase_estimation`` algorithm is used to perform the phase kickback and QFT
transformation. Note that this is computationally more
expensive than the original formulation by Jordan.

See the ``examples`` folder, ``JordanGradient.ipynb`` for more details on using this module.

.. [1] https://doi.org/10.1103/PhysRevLett.95.050501
