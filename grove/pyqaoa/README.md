# pyQAOA
A Python implementation of the Quantum Approximate Optimization Algorithm (QAOA) using
pyQuil.

## Structure

`qaoa.py` contains the base QAOA class and routines for finding the optimal
rotation angles via the variational-quantum-eigensolver (VQE) method, preparing states,
and storing results as well as utilities for probabilities and collecting bitstrings after a
state preparation.

`maxcut_qaoa.py` takes a graph defined with either NetworkX or a list of edges expressed as tuples
and implements the cost function for max-cut problems.

`numberpartiition_qaoa.py` takes a list of numbers and sets up a QAOA instance
for determining the equal biparitioning of the list.

## Run
 
The simplest way to interact with the QAOA library is through the methods provided for
each problem instance.  For example, to run max-cut import `maxcut_qaoa` from `maxcut_qaoa.py`
and pass a graph to the script.
This function will return a QAOA instance.  Calling `get_angles()` on the instance will
start the variational-quantum-eigensolver (VQE) loop in order to find the optimal beta and gamma
angles.

## Examples using QAOA

```
import numpy as np
from pyquil.api import WavefunctionSimulator

from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
```

```
steps = 2
square_ring = [(0,1),(1,2),(2,3),(3,0)]

inst = maxcut_qaoa(square_ring, steps=steps)
opt_betas, opt_gammas = inst.get_angles()
```

To see the final |beta,gamma> state we can rebuild the quil program that gives
us |beta,gamma> and evaluate the wavefunction.

```
t = np.hstack((opt_betas, opt_gammas))
param_prog = inst.get_parameterized_program()
prog = param_prog(t)
wf = WavefunctionSimulator().wavefunction(prog)
wf = wf.amplitudes
```

`wf` is now a numpy array of complex-valued amplitudes for each computational
basis state.  To visualize the distribution, iterate over the states and
calculate the probability.

```
for state_index in range(inst.nstates):
    print(inst.states[state_index], np.conj(wf[state_index]) * wf[state_index])
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
* pyQuil >= 2.0.0
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
directory and open the `index.html` file in a browser.
