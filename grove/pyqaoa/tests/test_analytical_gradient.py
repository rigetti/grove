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

import sys, os
#Redesign the relative import system
#Add the analytical gradient directory
dirname = os.path.dirname(os.path.abspath(__file__))
analytical_gradient_dir = os.path.join(dirname, '../analytical_gradient')
qaoa_dir = os.path.join(dirname, '..')
sys.path.append(analytical_gradient_dir)
sys.path.append(qaoa_dir)


import pyquil.quil as pq
from pyquil.gates import *
from pyquil.paulis import *

import maxcut_qaoa_core
import analytical_gradient
import utils

#What Graphs should I write test cases for?

def test_edges_to_graph():
    n_2_path_graph_edges = [(0,1)]
    n_2_path_graph = maxcut_qaoa_core.edges_to_graph(n_2_path_graph_edges)
    assert n_2_path_graph.nodes() == [0,1]

def test_reference_state_program():
    num_qubits = 2
    reference_state_program = \
        maxcut_qaoa_core.construct_reference_state_program(num_qubits)
    comparison_program = pq.Program().inst(H(0)).inst(H(1))
    utils.compare_progs(reference_state_program, comparison_program)

def test_get_cost_hamiltonian():
    n_2_path_graph_edges = [(0,1)]
    n_2_path_graph = maxcut_qaoa_core.edges_to_graph(n_2_path_graph_edges)
    cost_hamiltonian, cost_program = maxcut_qaoa_core.get_cost_hamiltonian(
        n_2_path_graph)
    comparison_hamiltonian = PauliSum([PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1)])
    comparison_program = pq.Program().inst(Z(0)).inst(Z(1))
    utils.compare_paulisums(cost_hamiltonian, comparison_hamiltonian)
    utils.compare_progs(cost_program, comparison_program)

def test_get_driver_hamiltonian():
    n_2_path_graph_edges = [(0,1)]
    n_2_path_graph = maxcut_qaoa_core.edges_to_graph(n_2_path_graph_edges)
    driver_hamiltonian, driver_program = maxcut_qaoa_core.get_driver_hamiltonian(
        n_2_path_graph)
    reference_hamiltonian = (PauliSum([PauliTerm("X", 0, 1.0)]) +
                             PauliSum([PauliTerm("X", 1, 1.0)]))
    utils.compare_paulisums(driver_hamiltonian, reference_hamiltonian)

def test_exponentiate_hamiltonian():
    n_2_path_graph_edges = [(0,1)]
    parameters = 0
    n_2_path_graph = maxcut_qaoa_core.edges_to_graph(n_2_path_graph_edges)
    cost_hamiltonian, cost_program = maxcut_qaoa_core.get_cost_hamiltonian(
        n_2_path_graph)
    exponentiate_hamiltonian(cost_hamiltonian)

if __name__ == "__main__":
    test_edges_to_graph()
    test_reference_state_program()
    test_get_cost_hamiltonian()
    test_get_driver_hamiltonian()
