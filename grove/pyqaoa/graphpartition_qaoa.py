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

from pyquil.paulis import PauliTerm, PauliSum
import numpy as np
import networkx as nx
import pyquil.forest as qvm_module
from scipy.optimize import minimize
from grove.pyqaoa.qaoa import QAOA
CXN = qvm_module.Connection()


def print_fun(x):
    print x


def graphpart_qaoa(graph, A=None, B=None, steps=1, minimizer_kwargs=None, rand_seed=42):

    if not isinstance(graph, nx.Graph) and isinstance(graph, list):
        if not all([isinstance(x, (tuple)) for x in graph]):
            raise TypeError("""List must contain ints, floats, longs""")

        maxcut_graph = nx.Graph()
        for edge in graph:
            maxcut_graph.add_edge(*edge)
        graph = maxcut_graph.copy()

    cost_operators = []
    driver_operators = []
    if A is None or B is None:
        # bounds from arXiv: 1302:5843v3
        B = 1
        max_degree = max(graph.degree(), key=lambda x: graph.degree()[x])
        A_B_ratio = min(2*max_degree, len(graph.nodes()))/8.0
        A = A_B_ratio
        print A, B

    for node in graph.nodes():
        for node_j in graph.nodes():
            cost_operators.append(PauliSum([PauliTerm("Z", node, A)*PauliTerm("Z", node_j, 1.0)]))
        driver_operators.append(PauliSum([PauliTerm("X", node, -1)]))

    for i, j in graph.edges():
        cost_operators.append(PauliTerm("Z", i, -B*0.5)*PauliTerm("Z", j, 1.0) + PauliTerm("I", 0, B*0.5))

    # set number of qubits
    n_qubits = len(graph.nodes())

    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': 1.0e-2,
                                        'xtol': 1.0e-1,
                                        'disp': True,
                                        'maxfev': 100}}

    qaoa_inst = QAOA(CXN, n_qubits, steps=steps, cost_ham=cost_operators,
                     ref_hamiltonian=driver_operators, store_basis=True,
                     minimizer=minimize,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options={'disp': None},
                     rand_seed=rand_seed)

    return qaoa_inst


if __name__ == "__main__":
    # Sample Run:
    # Cutting into equal partitions 0 - 1 - 2 - 3graph!
    inst = graphpart_qaoa([(0, 1), (1, 2), (2, 3)], steps=2)
    betas, gammas = inst.get_angles()
    probs = inst.probabilities(np.hstack((betas, gammas)))
    for state, prob in zip(inst.states, probs):
        print state, prob

    print "Most frequent bitstring from sampling"
    most_freq_string, sampling_results = inst.get_string(
            betas, gammas, samples=100)
    print most_freq_string
