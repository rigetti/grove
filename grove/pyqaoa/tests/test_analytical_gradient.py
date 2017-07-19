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
    hamiltonian_0 = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    hamiltonians = [hamiltonian_0]
    p_unitary_0 = maxcut_qaoa_core.exponential_map_hamiltonian(hamiltonian_0)
    p_unitaries = [p_unitary_0]
    make_controlled = analytical_gradient.generate_make_controlled(2)
    sum_of_branches = analytical_gradient.differentiate_product_rule(
        p_unitaries, hamiltonians, make_controlled)
    parameters = [0.1]
    evaluate_product = analytical_gradient.generate_evaluate_product(parameters)
    evaluated_sum = analytical_gradient.map_products(
        sum_of_branches, evaluate_product)
    comparison_sum = [[
        [pq.Program().inst(CNOT(2, 0)) + p_unitary_0(-parameters[0]/2)],
        [pq.Program().inst(CNOT(2, 1)) + p_unitary_0(-parameters[0]/2)]
        ]]
    assert len(comparison_sum) == len(evaluated_sum)
    for summand_idx in xrange(len(evaluated_sum)):
        assert (len(comparison_sum[summand_idx]) ==
                len(evaluated_sum[summand_idx]))
        for product_idx in xrange(len(evaluated_sum[summand_idx])):
            assert (len(comparison_sum[summand_idx][product_idx]) ==
                    len(evaluated_sum[summand_idx][product_idx]))
            for factor_idx in xrange(len(
                evaluated_sum[summand_idx][product_idx])):
                factor = evaluated_sum[summand_idx][product_idx][factor_idx]
                comparison_factor = comparison_sum[summand_idx][product_idx][factor_idx]
                utils.compare_progs(factor, comparison_factor)

def test_differentiate_product_rule_two_hams():
    hamiltonian_0 = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    hamiltonian_1 = PauliSum([PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1, 1.0)])
    hamiltonians = [hamiltonian_0, hamiltonian_1]
    p_unitaries = [maxcut_qaoa_core.exponential_map_hamiltonian(hamiltonian)
                   for hamiltonian in hamiltonians]
    make_controlled = analytical_gradient.generate_make_controlled(2)
    sum_of_products = analytical_gradient.differentiate_product_rule(
        p_unitaries, hamiltonians, make_controlled)
    parameters = [0.1, 0.7]
    evaluate_product = analytical_gradient.generate_evaluate_product(parameters)
    evaluated_sum = analytical_gradient.map_products(
        sum_of_products, evaluate_product)
    comparison_sum = [
        [
            [pq.Program().inst(CNOT(2, 0)) + p_unitaries[0](-parameters[0]/2),
             p_unitaries[1](-parameters[1]/2)],
            [pq.Program().inst(CNOT(2, 1)) + p_unitaries[0](-parameters[0]/2),
             p_unitaries[1](-parameters[1]/2)]
        ],
        [
            [p_unitaries[0](-parameters[0]/2),
             pq.Program().inst(CPHASE(np.pi)(2, 0), CPHASE(np.pi)(2, 1)) +
             p_unitaries[1](-parameters[1]/2)]
        ]
    ]
    assert len(comparison_sum) == len(evaluated_sum)
    for summand_idx in xrange(len(evaluated_sum)):
        assert (len(comparison_sum[summand_idx]) ==
                len(evaluated_sum[summand_idx]))
        for product_idx in xrange(len(evaluated_sum[summand_idx])):
            assert (len(comparison_sum[summand_idx][product_idx]) ==
                    len(evaluated_sum[summand_idx][product_idx]))
            for factor_idx in xrange(len(
                evaluated_sum[summand_idx][product_idx])):
                factor = evaluated_sum[summand_idx][product_idx][factor_idx]
                comparison_factor = comparison_sum[summand_idx][product_idx][factor_idx]
                utils.compare_progs(factor, comparison_factor)

def test_generate_analytical_gradient():
    driver_hamiltonian = PauliTerm("X", 0, 1.0) + PauliTerm("X", 1, 1.0)
    cost_hamiltonian = PauliSum([PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1, 1.0)])
    hamiltonians = [cost_hamiltonian, driver_hamiltonian]
    steps = 1
    num_qubits = 2
    qvm_connection = api.SyncConnection()
    gradient = analytical_gradient.generate_analytical_gradient(hamiltonians,
        cost_hamiltonian, qvm_connection, steps, num_qubits)
    beta = 0.1
    gamma = 0.2
    steps_parameters = [gamma, beta]
    gradient_values = gradient(steps_parameters)
    comparison_gradient_values = [np.sin(2*beta)*np.cos(gamma),
                                  2*np.cos(2*beta)*np.sin(gamma)]
    for gradient_idx in xrange(len(gradient_values)):
        assert utils.isclose(gradient_values[gradient_idx],
                             comparison_gradient_values[gradient_idx])

if __name__ == "__main__":
    test_generate_make_controlled()
    test_pauli_term_to_program()
    test_hamiltonian_to_program_branches()
    test_differentiate_unitary_sum()
    test_differentiate_unitary_product()
    test_parallelize()
    test_differentiate_product_rule_one_ham()
    test_differentiate_product_rule_two_hams()
    test_generate_analytical_gradient()
