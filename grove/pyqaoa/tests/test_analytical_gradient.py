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

import analytical_gradient
import maxcut_qaoa_core
import utils
import expectation_value


def test_generate_make_controlled():
    ancilla_qubit_index = 1
    make_controlled = analytical_gradient.generate_make_controlled(1)
    pauli_operator = [0, "X"]
    mapped_gate = make_controlled(pauli_operator)
    comparison_gate = CNOT(1,0)
    assert mapped_gate == comparison_gate

def test_pauli_term_to_program():
    pauli_term = PauliTerm("X", 0, 1.0)*PauliTerm("X", 1, 1.0)
    make_controlled = analytical_gradient.generate_make_controlled(2)
    program = analytical_gradient.pauli_term_to_program(pauli_term,
        make_controlled)
    comparison_program = pq.Program().inst(CNOT(2,0), CNOT(2,1))
    utils.compare_progs(program, comparison_program)

def test_hamiltonian_to_program_branches():
    hamiltonian = (PauliTerm("X", 0, 1.0)*PauliTerm("X", 1, 1.0) +
                   PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1, 1.0))
    make_controlled = analytical_gradient.generate_make_controlled(2)
    program_branches = analytical_gradient.hamiltonian_to_program_branches(
        hamiltonian, make_controlled)
    comparison_programs = [pq.Program().inst(CNOT(2, 0), CNOT(2, 1)),
        pq.Program().inst(CPHASE(np.pi)(2, 0), CPHASE(np.pi)(2, 1))]
    for idx in xrange(len(program_branches)):
        utils.compare_progs(program_branches[idx], comparison_programs[idx])

def test_differentiate_unitary_sum():
    generator = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    parameter = 1.0
    make_controlled = analytical_gradient.generate_make_controlled(2)
    p_unitary = maxcut_qaoa_core.exponential_map_hamiltonian(generator)
    unitary = p_unitary(parameter)
    p_analytical_derivative_branches = analytical_gradient.differentiate_unitary(
        p_unitary, generator, make_controlled)
    analytical_derivative_branches = map(lambda p_prog: p_prog(parameter),
        p_analytical_derivative_branches)
    comparison_programs = [pq.Program().inst(CNOT(2, 0)) + unitary,
                           pq.Program().inst(CNOT(2, 1)) + unitary]
    for idx in xrange(len(analytical_derivative_branches)):
        utils.compare_progs(analytical_derivative_branches[idx],
                                       comparison_programs[idx])

def test_differentiate_unitary_product():
    generator = PauliSum([PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1, 1.0)])
    parameter = 1.0
    make_controlled = analytical_gradient.generate_make_controlled(2)
    p_unitary = maxcut_qaoa_core.exponential_map_hamiltonian(generator)
    unitary = p_unitary(parameter)
    p_analytical_derivative_branches = analytical_gradient.differentiate_unitary(
        p_unitary, generator, make_controlled)
    analytical_derivative_branches = map(lambda p_prog: p_prog(parameter),
        p_analytical_derivative_branches)
    comparison_programs = [pq.Program().inst(CPHASE(np.pi)(2, 0),
        CPHASE(np.pi)(2, 1)) + unitary]
    for idx in xrange(len(analytical_derivative_branches)):
        utils.compare_progs(analytical_derivative_branches[idx],
                                       comparison_programs[idx])

def test_parallelize():
    column = [1,2,3]
    partial_row = [4,4,4]
    new_column_index = 1
    matrix = analytical_gradient.parallelize(column, partial_row,
        new_column_index)
    comparison_matrix = [[4, 1, 4], [4, 2, 4], [4, 3, 4]]
    assert matrix == comparison_matrix

def test_differentiate_product_rule_one_ham():
    #####################
    #For One Hamiltonian#
    #####################
    hamiltonian_0 = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    hamiltonians = [hamiltonian_0]
    p_unitary_0 = maxcut_qaoa_core.exponential_map_hamiltonian(hamiltonian_0)
    p_unitaries = [p_unitary_0]
    make_controlled = analytical_gradient.generate_make_controlled(2)
    differentiated_product = analytical_gradient.differentiate_product_rule(
        p_unitaries, hamiltonians, make_controlled)
    parameters = [0.1]
    evaluated_product = analytical_gradient.evaluate_differentiated_product(
        differentiated_product, parameters)
    comparison_product = [[
        [pq.Program().inst(CNOT(2, 0)) + p_unitary_0(parameters[0])],
        [pq.Program().inst(CNOT(2, 1)) + p_unitary_0(parameters[0])]
        ]]
    assert len(comparison_product) == len(evaluated_product)
    for summand_idx in xrange(len(evaluated_product)):
        #print(len(comparison_product[summand_idx]))
        #print(len(evaluated_product[summand_idx]))
        assert (len(comparison_product[summand_idx]) ==
                len(evaluated_product[summand_idx]))
        for factor_idx in xrange(len(evaluated_product[summand_idx])):
            #print(len(comparison_product[summand_idx][factor_idx]))
            #print(len(evaluated_product[summand_idx][factor_idx]))
            for branch_idx in xrange(len(
                    evaluated_product[summand_idx][factor_idx])):
                branch = evaluated_product[summand_idx][factor_idx][branch_idx]
                comparison_branch = comparison_product[summand_idx][factor_idx][branch_idx]
                utils.compare_progs(branch, comparison_branch)

def test_differentiate_product_rule_two_hams():
    ######################
    #For Two Hamiltonians#
    ######################
    hamiltonian_0 = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    hamiltonian_1 = PauliSum([PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1, 1.0)])
    hamiltonians = [hamiltonian_0, hamiltonian_1]
    p_unitaries = [maxcut_qaoa_core.exponential_map_hamiltonian(hamiltonian)
                   for hamiltonian in hamiltonians]
    make_controlled = analytical_gradient.generate_make_controlled(2)
    differentiated_product = analytical_gradient.differentiate_product_rule(
        p_unitaries, hamiltonians, make_controlled)
    parameters = [0.1, 0.7]
    evaluated_product = analytical_gradient.evaluate_differentiated_product(
        differentiated_product, parameters)
    comparison_product = [
        [
            [pq.Program().inst(CNOT(2, 0)) + p_unitary_0(parameters[0]),
             p_unitary_1(parameters[1])],
            [pq.Program().inst(CNOT(2, 1)) + p_unitary_0(parameters[0]),
             p_unitary_1(parameters[1])]
        ],
        [
            [p_unitary_0(parameters[0]),
             pq.Program().inst(CPHASE(np.pi)(2, 0), CPHASE(np.pi)(2, 1)) +
             p_unitary_1(parameters[1])]
        ],
    ]
    assert len(comparison_product) == len(evaluated_product)
    for summand_idx in xrange(len(evaluated_product)):
        #print(len(comparison_product[summand_idx]))
        #print(len(evaluated_product[summand_idx]))
        assert (len(comparison_product[summand_idx]) ==
                len(evaluated_product[summand_idx]))
        for factor_idx in xrange(len(evaluated_product[summand_idx])):
            #print(len(comparison_product[summand_idx][factor_idx]))
            #print(len(evaluated_product[summand_idx][factor_idx]))
            for branch_idx in xrange(len(
                    evaluated_product[summand_idx][factor_idx])):
                branch = evaluated_product[summand_idx][factor_idx][branch_idx]
                comparison_branch = comparison_product[summand_idx][factor_idx][branch_idx]
                utils.compare_progs(branch, comparison_branch)

def test_generate_analytical_gradient():
    hamiltonian_0 = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    hamiltonian_1 = PauliSum([PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1, 1.0)])
    hamiltonians = [hamiltonian_0, hamiltonian_1]
    make_controlled = analytical_gradient.generate_make_controlled(2)
    steps = 2
    analytical_gradient = analytical_gradient.generate_analytical_gradient(
        hamiltonians, make_controlled, steps)


def test_add_phase_correction():
    pass


def test_analytical_gradient_expectation_value():
    graph_edges = [(0,1)]
    steps = 1
    beta = 1.3
    gamma = 1.2
    parameters = [beta, gamma]

    gamma_derivative = np.sin(2*beta)*np.cos(gamma)
    beta_derivative = 2*np.cos(2*beta)*np.sin(gamma)

    graph = maxcut_qaoa_core.edges_to_graph(graph_edges)
    num_qubits = len(graph.nodes())
    ancilla_qubit_index = num_qubits
    reference_state_program = \
        maxcut_qaoa_core.construct_reference_state_program(num_qubits)
    cost_hamiltonian = maxcut_qaoa_core.get_cost_hamiltonian(graph)
    driver_hamiltonian = maxcut_qaoa_core.get_driver_hamiltonian(graph)
    make_controlled = analytical_gradient.generate_make_controlled(
        ancilla_qubit_index)
    analytical_gradient = analytical_gradient_generate_analytical_gradient(
        [cost_hamiltonian, driver_hamiltonian], make_controlled, steps)
    add_phase_correction(analytical_gradient, ancilla_qubit_index)

    for program in gradient_component_programs:
	print(program)
    full_cost_hamiltonian = extend_cost_hamiltonian(cost_hamiltonian,
	num_qubits)
    numerical_expectations = [expectation_value.expectation(
	gradient_component_program, full_cost_hamiltonian, qvm_connection)
	for gradient_component_program in gradient_component_programs]
    numerical_expectation = -sum(numerical_expectations)
    print(numerical_expectation)
    #driver_expectation = 2*np.cos(2*beta)*np.sin(gamma)
    #print(driver_expectation)
    print(cost_expectation)

    full_program = reference_state_program + maxcut_qaoa_unitary_program
    qvm_connection = api.SyncConnection()
    numerical_expectation_value = expectation_value.expectation(full_program,
        cost_hamiltonian, qvm_connection)

    analytical_expectation_value = np.sin(2*betas[0])*np.sin(gammas[0])
    assert round(analytical_expectation_value, 6) == round(numerical_expectation_value,6)


if __name__ == "__main__":
    test_generate_make_controlled()
    test_pauli_term_to_program()
    test_hamiltonian_to_program_branches()
    test_differentiate_unitary_sum()
    test_differentiate_unitary_product()
    test_parallelize()
    test_differentiate_product_rule_one_ham()
    test_differentiate_product_rule_two_hams()
    #test_unitaries_list_to_analytical_derivatives_list_branches_sum()
    #test_unitaries_list_to_analytical_derivatives_list_branches_product()
    #test_zip_analytical_derivatives_list_branches()
    #test_generate_step_analytical_gradient()
