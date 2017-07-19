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

import numpy as np
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import *

import maxcut_qaoa_core
import utils
import expectation_value

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
    cost_hamiltonian= maxcut_qaoa_core.get_cost_hamiltonian(n_2_path_graph)
    comparison_hamiltonian = PauliSum([PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1)])
    utils.compare_paulisums(cost_hamiltonian, comparison_hamiltonian)

def test_get_driver_hamiltonian():
    n_2_path_graph_edges = [(0,1)]
    n_2_path_graph = maxcut_qaoa_core.edges_to_graph(n_2_path_graph_edges)
    driver_hamiltonian = maxcut_qaoa_core.get_driver_hamiltonian(
        n_2_path_graph)
    reference_hamiltonian = (PauliSum([PauliTerm("X", 0, 1.0)]) +
                             PauliSum([PauliTerm("X", 1, 1.0)]))
    utils.compare_paulisums(driver_hamiltonian, reference_hamiltonian)

def test_exponential_map_hamiltonian():
    hamiltonian = PauliSum([PauliTerm("X", 0, 1.0)])
    parameter = 0.1
    p_unitary = maxcut_qaoa_core.exponential_map_hamiltonian(hamiltonian)
    unitary = p_unitary(parameter/2.0)
    comparison_unitary = pq.Program().inst(H(0), RZ(parameter)(0), H(0))
    utils.compare_progs(unitary, comparison_unitary)

def test_exponentiate_hamiltonian():
    hamiltonian = PauliSum([PauliTerm("X", 0, 1.0)])
    parameter = 0.1
    unitary = maxcut_qaoa_core.exponentiate_hamiltonian(hamiltonian,
        parameter/2.0)
    comparison_unitary = pq.Program().inst(H(0), RZ(parameter)(0), H(0))
    utils.compare_progs(unitary, comparison_unitary)

def test_get_program_parameterizer():
    steps = 2
    hamiltonian = PauliSum([PauliTerm("X", 0, 1.0)])
    program_parameterizer = maxcut_qaoa_core.get_program_parameterizer(steps,
        hamiltonian)
    angles = [0.1, np.pi]
    unitaries_list = program_parameterizer(angles)
    comparison_unitaries_list = [pq.Program().inst(H(0), RZ(angle)(0), H(0))
        for angle in angles]
    for idx in xrange(len(angles)):
        utils.compare_progs(unitaries_list[idx], comparison_unitaries_list[idx])

def test_zip_programs_lists():
    programs_list_A = [X(0), X(1)]
    programs_list_B = [Y(0), Y(1)]
    program = maxcut_qaoa_core.zip_programs_lists([programs_list_A,
                                                   programs_list_B])
    comparison_program = pq.Program().inst(X(0), Y(0), X(1), Y(1))
    utils.compare_progs(program, comparison_program)

def test_maxcut_qaoa_expectation_value():
    graph_edges = [(0,1)]
    steps = 1
    betas = [1.3]
    gammas = [1.2]

    graph = maxcut_qaoa_core.edges_to_graph(graph_edges)
    num_qubits = len(graph.nodes())
    reference_state_program = maxcut_qaoa_core.construct_reference_state_program(num_qubits)
    cost_hamiltonian = maxcut_qaoa_core.get_cost_hamiltonian(graph)
    driver_hamiltonian = maxcut_qaoa_core.get_driver_hamiltonian(graph)
    cost_unitary_list = maxcut_qaoa_core.get_program_parameterizer(
        steps, cost_hamiltonian)(gammas)
    driver_unitary_list = maxcut_qaoa_core.get_program_parameterizer(
        steps, driver_hamiltonian)(betas)
    maxcut_qaoa_unitary_program = maxcut_qaoa_core.zip_programs_lists(
        [cost_unitary_list, driver_unitary_list])
    full_program = reference_state_program + maxcut_qaoa_unitary_program
    qvm_connection = api.SyncConnection()
    numerical_expectation_value = expectation_value.expectation(full_program,
        cost_hamiltonian, qvm_connection)

    analytical_expectation_value = np.sin(2*betas[0])*np.sin(gammas[0])
    assert round(analytical_expectation_value, 6) == round(numerical_expectation_value,6)


if __name__ == "__main__":
    test_edges_to_graph()
    test_reference_state_program()
    test_get_cost_hamiltonian()
    test_get_driver_hamiltonian()
    test_exponential_map_hamiltonian()
    test_exponentiate_hamiltonian()
    test_get_program_parameterizer()
    test_zip_programs_lists()
    test_maxcut_qaoa_expectation_value()
