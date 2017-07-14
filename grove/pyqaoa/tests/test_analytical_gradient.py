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

def test_get_analytical_derivative_unitary_product():
    generator = PauliSum([PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1, 1.0)])
    parameter = 1.0
    make_controlled = analytical_gradient.generate_make_controlled(2)
    unitary = maxcut_qaoa_core.exponentiate_hamiltonian(generator, parameter)
    analytical_derivative_branches = \
        analytical_gradient.get_analytical_derivative_branches(
        unitary, generator, make_controlled)
    comparison_programs = [pq.Program().inst(CPHASE(np.pi)(2, 0),
        CPHASE(np.pi)(2, 1)) + unitary]
    for idx in xrange(len(analytical_derivative_branches)):
        utils.compare_progs(analytical_derivative_branches[idx],
                                       comparison_programs[idx])

def test_get_analytical_derivative_unitary_sum():
    generator = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    parameter = 1.0
    make_controlled = analytical_gradient.generate_make_controlled(2)
    unitary = maxcut_qaoa_core.exponentiate_hamiltonian(generator, parameter)
    analytical_derivative_branches = \
        analytical_gradient.get_analytical_derivative_branches(
        unitary, generator, make_controlled)
    comparison_programs = [pq.Program().inst(CNOT(2, 0)) + unitary,
                           pq.Program().inst(CNOT(2, 1)) + unitary]
    for idx in xrange(len(analytical_derivative_branches)):
        utils.compare_progs(analytical_derivative_branches[idx],
                                       comparison_programs[idx])

def test_unitaries_list_to_analytical_derivatives_list_branches():
    step_index = 0
    hamiltonian = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    unitary_step_0 = maxcut_qaoa_core.exponentiate_hamiltonian(hamiltonian, 0.1)
    unitary_step_1 = maxcut_qaoa_core.exponentiate_hamiltonian(hamiltonian, 0.7)
    unitaries_list = [unitary_step_0, unitary_step_1]
    make_controlled = analytical_gradient.generate_make_controlled(2)
    analytical_derivatives_list_branches = \
        analytical_gradient.unitaries_list_to_analytical_derivatives_list_branches(
        step_index, hamiltonian, unitaries_list, make_controlled)
    comparison_programs_list_branch_A = [
        pq.Program().inst(CNOT(2, 0)) + unitary_step_0, unitary_step_1]
    comparison_programs_list_branch_B = [
        pq.Program().inst(CNOT(2, 1)) + unitary_step_0, unitary_step_1]
    comparison_programs_list_branches = [comparison_programs_list_branch_A,
                                         comparison_programs_list_branch_B]
    assert (len(analytical_derivatives_list_branches) ==
            len(comparison_programs_list_branches))
    for idx in xrange(len(analytical_derivatives_list_branches)):
        analytical_derivatives_list_branch = \
            analytical_derivatives_list_branches[idx]
        comparison_programs_list_branch = comparison_programs_list_branches[idx]
        assert (len(analytical_derivatives_list_branch) ==
                len(comparison_programs_list_branch))
        for jdx in xrange(len(analytical_derivatives_list_branch)):
            #print(analytical_derivatives_list_branch[jdx])
            #print(comparison_programs_list_branch[jdx])
            utils.compare_progs(analytical_derivatives_list_branch[jdx],
                                comparison_programs_list_branch[jdx])

def test_generate_step_analytical_gradient():
    hamiltonian_A = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    hamiltonian_B = PauliSum([PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1, 1.0)])
    hamiltonians = [hamiltonian_A, hamiltonian_B]
    unitaries_A_list = [
        maxcut_qaoa_core.exponentiate_hamiltonian(hamiltonian_A, 0.1),
        maxcut_qaoa_core.exponentiate_hamiltonian(hamiltonian_A, 0.7)]
    unitaries_B_list = [
        maxcut_qaoa_core.exponentiate_hamiltonian(hamiltonian_B, 0.3),
        maxcut_qaoa_core.exponentiate_hamiltonian(hamiltonian_B, 0.8)]
    unitaries_lists = [unitaries_A_list, unitaries_B_list]
    make_controlled = analytical_gradient.generate_make_controlled(2)
    step_analytical_gradient = \
        analytical_gradient.generate_step_analytical_gradient(
        unitaries_lists, hamiltonians, make_controlled)
    step_0_analytical_gradient = step_analytical_gradient(0)
    print(step_0_analytical_gradient)


def test_analytical_gradient_expectation_value():
    graph_edges = [(0,1)]
    steps = 1
    betas = [1.3]
    gammas = [1.2]

    graph = maxcut_qaoa_core.edges_to_graph(graph_edges)
    num_qubits = len(graph.nodes())
    ancilla_qubit_index = num_qubits
    reference_state_program = \
        maxcut_qaoa_core.construct_reference_state_program(num_qubits)
    cost_hamiltonian = maxcut_qaoa_core.get_cost_hamiltonian(graph)
    driver_hamiltonian = maxcut_qaoa_core.get_driver_hamiltonian(graph)
    cost_unitary_list = maxcut_qaoa_core.get_program_parameterizer(
        steps, cost_hamiltonian)(gammas)
    driver_unitary_list = maxcut_qaoa_core.get_program_parameterizer(
        steps, driver_hamiltonian)(betas)
    make_controlled = analytical_gradient.generate_make_controlled(
        ancilla_qubit_index)
    step_analytical_gradient = generate_step_analytical_gradient(
        [cost_unitary_list, driver_unitary_list],
        [cost_hamiltonian, driver_hamiltonian], make_controlled)


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
    cost_expectation = np.sin(2*beta)*np.cos(gamma)
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
    test_get_analytical_derivative_unitary_product()
    test_get_analytical_derivative_unitary_sum()
    test_unitaries_list_to_analytical_derivatives_list_branches()
    test_generate_step_analytical_gradient()
