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

"""
Finding a maximum cut by QAOA.
"""
import numpy as np
import pyquil.forest as qvm_module
from pyquil.paulis import PauliTerm, PauliSum
import networkx as nx
from scipy.optimize import minimize
from grove.pyqaoa.qaoa import QAOA
CXN = qvm_module.Connection()


def print_fun(x):
    print x


def maxcut_qaoa(graph, steps=1, rand_seed=None):
    """
    Max cut set up method
    """
    if not isinstance(graph, nx.Graph) and isinstance(graph, list):
        maxcut_graph = nx.Graph()
        for edge in graph:
            maxcut_graph.add_edge(*edge)
        graph = maxcut_graph.copy()

    cost_operators = []
    driver_operators = []
    for i, j in graph.edges():
        cost_operators.append(PauliTerm("Z", i, 0.5)*PauliTerm("Z", j) + PauliTerm("I", 0, -0.5))
    for i in graph.nodes():
        driver_operators.append(PauliSum([PauliTerm("X", i, -1.0)]))

    qaoa_inst = QAOA(CXN, len(graph.nodes()), steps=steps, cost_ham=cost_operators,
                     ref_hamiltonian=driver_operators, store_basis=True,
                     rand_seed=rand_seed,
                     minimizer=minimize,
                     minimizer_kwargs={'method': 'Nelder-Mead',
                                       'options': {'ftol': 1.0e-2,
                                                   'xtol': 1.0e-2,
                                                   'disp': False}},
                     vqe_options={'disp': print_fun, 'return_all': True})

    return qaoa_inst


if __name__ == "__main__":
    # Sample Run:
    # Cutting 0 - 1 - 2 graph!
    inst = maxcut_qaoa([(0, 1), (1, 2)],
                       steps=2, rand_seed=42)
    betas, gammas = inst.get_angles()
    probs = inst.probabilities(np.hstack((betas, gammas)))
    for state, prob in zip(inst.states, probs):
        print state, prob

    print "Most frequent bitstring from sampling"
    most_freq_string, sampling_results = inst.get_string(
            betas, gammas, samples=100)
    print most_freq_string
