##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

from __future__ import print_function
from pyquil.paulis import PauliTerm
import numpy as np
import networkx as nx
import pyquil.forest as qvm_module
import pyquil.quil as pq
from pyquil.gates import X
from scipy.optimize import minimize
from grove.pyqaoa.qaoa import QAOA
CXN = qvm_module.Connection()


def graphpart_jc_qaoa(graph, steps=1, minimizer_kwargs=None, rand_seed=42):

    if not isinstance(graph, nx.Graph) and isinstance(graph, list):
        if not all([isinstance(x, (tuple)) for x in graph]):
            raise TypeError("""List must contain ints, floats, longs""")

        maxcut_graph = nx.Graph()
        for edge in graph:
            maxcut_graph.add_edge(*edge)
        graph = maxcut_graph.copy()

    cost_operators = []
    driver_operators = []

    set_length = len(graph.nodes())
    for node in xrange(set_length):
        one_driver_term = PauliTerm("X", node, -1.0) * PauliTerm("X", (node + 1) % set_length, 1.0)
        one_driver_term = one_driver_term + PauliTerm("Y", node, -1.0) * PauliTerm("Y", (node + 1) % set_length, 1.0)
        driver_operators.append(one_driver_term)

    for i, j in graph.edges():
        cost_operators.append(PauliTerm("Z", i, -0.5)*PauliTerm("Z", j, 1.0) + PauliTerm("I", 0, 0.5))

    # set number of qubits
    n_qubits = len(graph.nodes())

    # prepare driver initial state program
    driver_init_prog = pq.Program()
    for x in xrange(n_qubits/2):
        driver_init_prog.inst(X(x))

    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': 1.0e-2,
                                        'xtol': 1.0e-1,
                                        'disp': True,
                                        'maxfev': 100}}

    qaoa_inst = QAOA(CXN, n_qubits, steps=steps, cost_ham=cost_operators,
                     ref_hamiltonian=driver_operators, store_basis=True,
                     minimizer=minimize,
                     driver_ref=driver_init_prog,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options={'disp': print},
                     rand_seed=rand_seed)

    return qaoa_inst


if __name__ == "__main__":
    # Sample Run:
    # Cutting 0 - 1 - 2 - 3 graph!
    inst = graphpart_jc_qaoa([(0, 1), (1, 2), (2, 3)], steps=1)
    betas, gammas = inst.get_angles()
    print("rotation angles ")
    print("betas ")
    print(betas)
    print("gammas ")
    print(gammas)
    print("---------------------------")
    probs = inst.probabilities(np.hstack((betas, gammas)))
    for state, prob in zip(inst.states, probs):
        print(state, prob)

    print("Most frequent bitstring from sampling")
    most_freq_string, sampling_results = inst.get_string(
            betas, gammas, samples=100)
    print(most_freq_string)
