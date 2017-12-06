# pyqaoa
A python implementation of the Quantum Approximate Optimization Algorithm using
pyQuil and the Rigetti Forest.

## Structure

`qaoa.py` contains the base QAOA class and routines for finding the optimal
rotation angles via the variational-quantum-eigensolver method, state
preparation methods, storing results, and utilities for probabilities and
collecting bitstrings after a state preparation.

`maxcut_qaoa.py` takes a graph defined with either NetworkX or a list of node
pairs and implements the cost function for MAX-CUT problems.

`numberpartiition_qaoa.py` takes a list of numbers and set sup a QAOA instance
for determining the equal biparitioning of the list.

## Run
 
The simplest way to interact with the QAOA library is through the methods provided for each problem instance.  For example, to run max cut import `maxcut_qaoa` from `maxcut_qaoa.py` and pass graph to the script. 
This function will return a QAOA instance.  Calling `get_angles()` on the instance will
start the variational-quantum-eigensolver loop in order to find  the beta, gamma angles.

## Examples using qaoa

```
import numpy as np
from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
import pyquil.api as api
qvm_connection = api.QVMConnection()
```

```
square_ring = [(0,1),(1,2),(2,3),(3,0)]
steps = 2; n_qubits = 4
betas = np.random.uniform(0, np.pi, p); gammas = np.random.uniform(0, 2*np.pi, p)
inst = maxcut_qaoa(square_ring, steps=steps)
inst.get_angles()
```

to see the final |beta,gamma> state we can rebuild the quil program that gives
us |beta,gamma> and evaluate the wave function using the **qvm**

```
t = np.hstack((inst.betas, inst.gammas))
param_prog = inst.get_parameterized_program()
prog = param_prog(t)
wf = qvm_connection.wavefunction(prog)
wf = wf.amplitudes
```

`wf` is now a numpy array of complex-valued amplitudes for each computational
basis state.  To visualize the distribution iterate over the states and
calculate the probability.

```
for state_index in range(2**inst.n_qubits):
    print inst.states[state_index], np.conj(wf[state_index])*wf[state_index]
```

You should then see that the algorithm converges on the expected solutions of 0101 and 1010!

```
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
```

Dependencies
------------

* Numpy
* Scipy
* pyQuil
* Mock (for development testing)
* NetworkX (for building and analyzing graphs)
* Matplotlib (useful for plotting)

Building the Docs
------------
To build the documentation run
```
cd docs/
make html
```

To view the docs navigate to the `docs/_build` directory in the pyQAOA root
directory and open the index.html file a browser. 
