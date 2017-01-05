===========================
Installation and Quickstart
===========================

Dependencies
------------

pyQAOA depends on a few scientific python packages as well as the python library for Quil:

* numpy
* scipy
* `pyQuil <https://github.com/rigetticomputing/pyQuil.git>`_
* NetworkX
* Matplotlib

Optional

* pytest (for testing)

**Numpy** and **Scipy** can be installed with `pip` (a python package manager). ::

    pip install numpy
    pip install scipy

Or using the Conda package manager ::

    conda install numpy
    conda install scipy

**pyQuil** can be installed by changing directory to where you would like to keep
the pyQuil repository and running ::

    git clone https://github.com/rigetticomputing/pyQuil.git
    cd pyQuil
    pip install -e .


You will need to make sure that your pyQuil installation is properly configured to run with a
QVM or quantum processor. See the pyQuil documentation for instructions on how to do this.


pyQAOA Installation
-------------------

Clone the `git repository <https://github.com/rigetticomputing/pyQAOA.git>`_, `cd` into it, and
run ::

   pip install -e . 

Quick Start Examples
--------------------

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

Then we specify the basic configuration parameters for the algorithm. This includes the number of
steps to use (which loosely corresponds to the accuracy of the optimization computation), the
number of required qubits, and the starting optimization parameters (betas and gammas):

.. code-block:: python

    steps = 2; n_qubits = 4
    betas = np.random.uniform(0, np.pi, steps); gammas = np.random.uniform(0, 2*np.pi, steps)

Now we instantiate the algorithm and run the optimization routine on our QVM:

.. code-block:: python

    inst = maxcut_qaoa(graph=square_ring, steps=steps, init_betas=betas, init_gammas=gammas)
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
