Jordan Gradient
==========================

Overview
--------

This is an implementation of Stephen Jordan's "Fast Quantum Algorithm for
Numerical Gradient Estimation" [1]_. It utilizes an oracle to sample some
function at some distance, h, from the point of evaluation. These distances are
encoded into the phase of an unitary operator. The `phase_estimation` algorithm
is then used to find this phase, which to a first order approximation, will
give you the gradient of your function.

See the `examples` folder, `JordanGradient.ipynb` for more details on using this module.

.. [1] https://doi.org/10.1103/PhysRevLett.95.050501
