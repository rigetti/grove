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
import pyquil.api as api
from pyquil.paulis import PauliTerm, PauliSum
import networkx as nx
from scipy.optimize import minimize
from grove.pyqaoa.qaoa import QAOA
CXN = api.SyncConnection()


def print_fun(x):
    print x


def maxcut_qaoa(graph, steps=1, rand_seed=None, connection=None, samples=None,
                initial_beta=None, initial_gamma=None, minimizer_kwargs=None,
                vqe_option=None):
    """
    Max cut set up method

    :param graph: Graph definition. Either networkx or list of tuples
    :param steps: (Optional. Default=1) Trotterization order for the
                  QAOA algorithm.
    :param rand_seed: (Optional. Default=None) random seed when beta and
                      gamma angles are not provided.
    :param connection: (Optional) connection to the QVM. Default is None.
    :param samples: (Optional. Default=None) VQE option. Number of samples
                    (circuit preparation and measurement) to use in operator
                    averaging.
    :param initial_beta: (Optional. Default=None) Initial guess for beta
                         parameters.
    :param initial_gamma: (Optional. Default=None) Initial guess for gamma
                          parameters.
    :param minimizer_kwargs: (Optional. Default=None). Minimizer optional
                             arguments.  If None set to
                             {'method': 'Nelder-Mead',
                             'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                        'disp': False}
    :param vqe_option: (Optional. Default=None). VQE optional
                             arguments.  If None set to
                       vqe_option = {'disp': print_fun, 'return_all': True,
                       'samples': samples}

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

    if connection is None:
        connection = CXN

    if minimizer_kwargs is None:
        minimizer_kwargs = {'method': 'Nelder-Mead',
                            'options': {'ftol': 1.0e-2, 'xtol': 1.0e-2,
                                        'disp': False}}
    if vqe_option is None:
        vqe_option = {'disp': print_fun, 'return_all': True,
                      'samples': samples}

    qaoa_inst = QAOA(connection, len(graph.nodes()), steps=steps, cost_ham=cost_operators,
                     ref_hamiltonian=driver_operators, store_basis=True,
                     rand_seed=rand_seed,
                     init_betas=initial_beta,
                     init_gammas=initial_gamma,
                     minimizer=minimize,
                     minimizer_kwargs=minimizer_kwargs,
                     vqe_options=vqe_option)

    return qaoa_inst


if __name__ == "__main__":
    # Sample Run:
    # Cutting 0 - 1 - 2 graph!
    inst = maxcut_qaoa([(0, 1), (1, 2)],
                       steps=2, rand_seed=42, samples=None)
    betas, gammas = inst.get_angles()
    probs = inst.probabilities(np.hstack((betas, gammas)))
    for state, prob in zip(inst.states, probs):
        print state, prob

    print "Most frequent bitstring from sampling"
    most_freq_string, sampling_results = inst.get_string(
            betas, gammas)
    print most_freq_string
