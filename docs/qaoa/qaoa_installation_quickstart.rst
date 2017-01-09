====================
Quick Start Examples
====================

To test your installation and get going we can run QAOA to solve MAX-CUT on a square ring with
4 nodes at the corners. In your python interpreter import the packages and connect to your QVM:

.. code-block:: python

    import numpy as np
    from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
    import pyquil.forest as qvm
    qvm_connection = qvm.Connection()

Next define the graph on which to run MAX-CUT

.. code-block:: python

    square_ring = [(0,1),(1,2),(2,3),(3,0)]

The optional configuration parameter for the algorithm is given by the number of
steps to use (which loosely corresponds to the accuracy of the optimization computation).
We instantiate the algorithm and run the optimization routine on our QVM:

.. code-block:: python

    steps = 2
    inst = maxcut_qaoa(graph=square_ring, steps=steps)
    inst.get_angles()

to see the final \\(\\mid \\beta, \\gamma \\rangle \\) state we can rebuild the
quil program that gives us \\(\\mid \\beta, \\gamma \\rangle \\)  and evaluate the wave function using the **qvm**

.. code-block:: python

    t = np.hstack((inst.betas, inst.gammas))
    param_prog = inst.get_parameterized_program()
    prog = param_prog(t)
    wf = qvm_connection.wavefunction(prog)

``wf`` is now a numpy array of complex-valued amplitudes for each computational
basis state.  To visualize the distribution iterate over the states and
calculate the probability.

.. code-block:: python

    for state_index in range(2**inst.n_qubits):
        print inst.states[state_index], np.conj(wf[state_index])*wf[state_index]

You should then see that the algorithm converges on the expected solutions of 0101 and 1010! ::

    0000 (4.38395094039e-26+0j)
    0001 (5.26193287055e-15+0j)
    0010 (5.2619328789e-15+0j)
    0011 (1.52416449345e-13+0j)
    0100 (5.26193285935e-15+0j)
    0101 (0.5+0j)
    0110 (1.52416449362e-13+0j)
    0111 (5.26193286607e-15+0j)
    1000 (5.26193286607e-15+0j)
    1001 (1.52416449362e-13+0j)
    1010 (0.5+0j)
    1011 (5.26193285935e-15+0j)
    1100 (1.52416449345e-13+0j)
    1101 (5.2619328789e-15+0j)
    1110 (5.26193287055e-15+0j)
    1111 (4.38395094039e-26+0j)
