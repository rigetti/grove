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

`numberpartiition_qaoa.py` takes a list of numbers and sets up a QAOA instance
for determining the equal biparitioning of the list.

## Run
 
The simplest way to interact with the QAOA library is through the methods provided for each problem instance.  For example, to run max cut import `maxcut_qaoa` from `maxcut_qaoa.py` and pass graph to the script. 
This function will return a QAOA instance.  Calling `get_angles()` on the instance will
start the variational-quantum-eigensolver loop in order to find  the beta, gamma angles.

## Examples using qaoa

```
import numpy as np
from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
```

```
#The square_ring is a list of the edges in the graph
#Where each edge is given as a tuple: (node1_index, node2_index)
square_ring = [(0,1),(1,2),(2,3),(3,0)]
#The number of trotterization steps is also referred to as "p".
trotterization_steps = 2
#maxcut_qaoa constructs the sequence of parametric gates
inst = maxcut_qaoa(square_ring, steps=trotterization_steps)
#get_angles() finds the optimal beta and gamma angles
betas, gammas = inst.get_angles()
```
Using the sequence of gates which produced the optimal |beta, gamma> state,
the algorithm constructs the wave function via connecting to the qvm.
From the amplitudes of this wave function, the algorithm produces the
probabilities of any particular cut of the given graph. The cuts with the 
maximum probabilities are the solutions.

```
probs = inst.probabilities(np.hstack((betas, gammas)))
```

Each cut is represented as a tuple of 0s and 1s where the nodes at the
0 indices are in one partition and the nodes at the 1 indices are in the
other partition.

To visualize the distribution of probabilities for all the various graph cuts
we iterate over the cuts and display the corresponding probabilities.

```
for state, prob in zip(inst.states, probs):
     print state, prob
```

In particular, for our square graph example you should then see that the 
algorithm converges on the expected solutions of 0101 and 1010!

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
