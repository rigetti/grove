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
import analytical_gradient
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
    hamiltonian = PauliSum([PauliTerm("X", 0, 1.0)])
    parameter = 0.1
    unitary = maxcut_qaoa_core.exponentiate_hamiltonian(hamiltonian,
        parameter/2.0)
    comparison_unitary = pq.Program().inst(H(0), RZ(parameter)(0), H(0))
    utils.compare_progs(unitary, comparison_unitary)

def test_pauli_term_to_gate():
    pauli_term = PauliTerm("X", 0, 1.0)
    gate = maxcut_qaoa_core.pauli_term_to_gate(pauli_term)
    comparison_gate = X(0)
    assert gate == comparison_gate

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

def test_maxcut_qaoa_expectation_value():
    graph_edges = [(0,1)]
    steps = 1
    beta = 1.3
    gamma = 1.2
    angles = [beta, gamma]

    program_parameterizer, reference_state_program, cost_hamiltonian, num_qubits = \
	maxcut_qaoa_core.maxcut_qaoa_constructor(graph_edges, steps)
    parameterized_program = program_parameterizer(angles)

    full_program = reference_state_program + parameterized_program
    qvm_connection = api.SyncConnection()
    numerical_expectation_value = expectation_value.expectation(full_program,
        cost_hamiltonian, qvm_connection)

    analytical_expectation_value = np.sin(2*beta)*np.sin(gamma)
    assert round(analytical_expectation_value, 6) == round(numerical_expectation_value,6)

if __name__ == "__main__":
    test_edges_to_graph()
    test_reference_state_program()
    test_get_cost_hamiltonian()
    test_get_driver_hamiltonian()
    test_exponentiate_hamiltonian()
    test_pauli_term_to_gate()
    test_get_program_parameterizer()
    test_maxcut_qaoa_expectation_value()
