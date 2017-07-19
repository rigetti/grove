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
This module implements the analytical gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
"""


import numpy as np
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import *

import maxcut_qaoa_core
import expectation_value


def generate_make_controlled(ancilla_qubit_index):
    """
    Creates a function which maps operators to controlled operators
    :param (int) ancilla_qubit_index: controls the controlled gates
    :return (function) make_controlled: maps gates to controlled gates
    """

    def make_controlled(pauli_operator):
        """
        Maps Pauli operators to the corresponding controlled gate
        :param (pauli_op) pauli_operator: a qubit_index, and gate_name pair
        :return (Gate) mapped_gate: the corresponding controlled gate
        """
        qubit_index, gate_name = pauli_operator
        mapping_dict = {
            "X": CNOT,
            "Z": CPHASE(np.pi)
            }
        mapped_gate_op = mapping_dict[gate_name]
        qubit_indices = [ancilla_qubit_index, qubit_index]
        mapped_gate = mapped_gate_op(*qubit_indices)
        return mapped_gate

    return make_controlled

def pauli_term_to_program(pauli_term, pauli_operator_mapping):
    """
    Converts a PauliTerm to the corresponding list of Gates
    :param (PauliTerm) pauli_term: has a list of pauli operators
    :param (function) operator_mapping: e.g. make_controlled
    :return (list[Gate]) program: lookup each gate in the STANDARD_GATES dict
    """
    program = pq.Program()
    for pauli_operator in pauli_term:
        mapped_gate = pauli_operator_mapping(pauli_operator)
        program.inst(mapped_gate)
    return program

def hamiltonian_to_program_branches(hamiltonian, pauli_operator_mapping):
    """
    Converts a hamiltonian to the corresponding list of programs
    :param (PauliSum) hamiltonian: has a list of pauli terms
    :return (list[pq.Program]) program_branches: each branch is from a term
    """
    program_branches = [pauli_term_to_program(pauli_term, pauli_operator_mapping)
        for pauli_term in hamiltonian]
    return program_branches

def gen_p_analytical_derivative(program_branch, p_unitary):
    """
    Creates one branch of an analytical derivative
    :param (pq.Program) program_branch: from one term of the generator
    :param (function) p_unitary: the differentiated unitary
    :return (function) p_analytical_derivative_branch: for the program branch
    """
    def p_analytical_derivative_branch(param):
        return program_branch + p_unitary(param)
    return p_analytical_derivative_branch

def differentiate_unitary(p_unitary, generator,
        pauli_operator_mapping):
    """
    Writes the parameterized analytical derivative of a unitary
    :param (function) p_unitary: the p_unitary to be differentiated
    :param (PauliSum) generator: generates the p_unitary
    :param (function) pauli_operator_mapping:
    :return (list[function]) p_analytical_derivative_branches: for each term
    """
    program_branches = hamiltonian_to_program_branches(generator,
        pauli_operator_mapping)
    p_analytical_derivative_branches = []
    for program_branch in program_branches:
        p_analytical_derivative_branches.append(
            gen_p_analytical_derivative(program_branch, p_unitary))
    return p_analytical_derivative_branches

def parallelize(new_column, old_row, new_column_index):
    """
    Map a column and a row to a full matrix
    :param (list[Abstract]) new_column: the new elements to be added
    :param (list[Abstract]) old_row: replace elements at new_column_index
    :param (int) new_column_index: the index in old_row to be replaced
    :return (list[list[Abstract]]) matrix: [column][row]
    """
    matrix = []
    for column_element in new_column:
        new_row = [column_element if column_index == new_column_index
                   else row_element for column_index, row_element
                   in enumerate(old_row)]
        matrix.append(new_row)
    return matrix

def differentiate_product_rule(p_unitaries_product, hamiltonians_list,
        make_controlled):
    """
    Differentiates a product of parametric unitariesusing the product rule
    :param (list[function]) p_unitaries_product: [uni] i.e. unitary_index
    :param (list[PauliSum]) hamiltonians_list: for each factor in the product
    :param (function) make_controlled: e.g. X -> CNOT
    :return (list[list[list[function]]]) sum_of_branches: [summand][branch][uni]
    """
    sum_of_branches = []
    for unitary_index in xrange(len(p_unitaries_product)):
        p_analytical_derivative_branches = differentiate_unitary(
            p_unitaries_product[unitary_index],
            hamiltonians_list[unitary_index], make_controlled)
        branches_of_products = parallelize(p_analytical_derivative_branches,
                p_unitaries_product, unitary_index)
        sum_of_branches.append(branches_of_products)
    return sum_of_branches

###########################################
#An aside on the sum_of_branches structure#
###########################################
#The original program is of the form e^{-ia/2(G_1 + G_2)} e^{-ib/2(G_3)}
#where the G_i are the Hermitian generators of the Unitaries
#This structure is a list of parametric unitaries, with one index
#Taking the derivaative of this original program results in:
#D(e^{-ia/2(G_1 + G_2)} e^{-ib/2(G_3)})
#Which by the product rule can be simplified as:
#D(e^{-ia/2(G_1 + G_2)}) e^{-ib/2(G_3)} + e^{-ia/2(G_1 + G_2)} D(e^{-ib/2(G_3)})
#Now we have gained an additional index, over the summands
#and so our structure is now of the form (list[summand_idx][unitary_idx])
#Now we further simplify to find:
#(-i/2)[(G_1 + G_2)e^{-ia/2(G_1 + G_2)}) e^{-ib/2(G_3)} +
#        e^{-ia/2(G_1 + G_2)} e^{-ib/2(G_3)}]
#So now we must index over each generator in each summand
#and the structure is of the form (list[summand_idx][branch_idx][unitary_idx])

def generate_evaluate_product(parameters):
    """
    Creates the function which evaluates a product of parameterized programs
    :param (list[float]) parameters: for evaluating a product of factors
    :return (function) evaluate_product: contains the factor of two correction
    """
    def evaluate_product(product):
        """
        Evaluates each factor in the product on the corresponding parameter
        :param (list[function]) product: parameterized programs
        :return (list[pq.Program]) evaluated_product: programs
        """
        assert len(product) == len(parameters)
        evaluated_product = []
        for factor, parameter in zip(product, parameters):
            evaluated_factor = factor(-parameter/2)
            evaluated_product.append(evaluated_factor)
        return evaluated_product
    return evaluate_product

def map_branches(sum_of_branches, branch_map):
    """
    Map each product in the sum of products using the given map
    :param (list[list[list[Abstract_1]]]) sum_of_branches: see below
    :param (function: list[list[Abstract_1]] -> Abstract_2) product_map:
    :return (list[Abstract_2]) mapped_sum:
    """
    mapped_sum = []
    for branch in sum_of_branches:
        mapped_branch = branch_map(branch)
        mapped_sum.append(mapped_branch)
    return mapped_sum

def map_products(sum_of_branches, product_map):
    """
    Map each product in the sum of products using the given map
    :param (list[list[list[Abstract_1]]]) sum_of_branches:
    :param (function: list[Abstract_1] -> Abstract_2) product_map:
    :return (list[list[Abstract_2]]) mapped_sum:
    """
    mapped_sum = []
    for branch in sum_of_branches:
        mapped_branch = []
        for product in branch:
            mapped_product = product_map(product)
            mapped_branch.append(mapped_product)
        mapped_sum.append(mapped_branch)
    return mapped_sum

def map_factors(sum_of_products, factor_map):
    """
    Map each factor in each of the products using the given map
    :param (list[list[list[Abstract_1]]]) sum_of_products:
    :param (function: Abstract_1 -> Abstract_2) factor_map:
    :return (list[list[list[Abstract_2]]) mapped_sum:
    """
    mapped_sum = []
    for branch in sum_of_branches:
        mapped_branch = []
        for product in branch:
            mapped_product = []
            for factor in product:
                mapped_factor = factor_map(factor)
                mapped_product.append(mapped_factor)
            mapped_branch.append(mapped_product)
        mapped_sum.append(mapped_branch)
    return mapped_sum

def generate_state_preparation(num_qubits):
    """
    Generates function for prepending state preparation gates
    :param (int) num_qubits: Hadamard each qubit
    :return (function) add_state_preparation: superposition of all bitstrings
    """
    def add_state_preparation(gradient_term):
        """
        Prepends state preparation gates to a given term in the gradient
        :param (pq.Program) gradient_term: Add hadamard gates before the program
        :return (pq.Program) prepared_term: program with state preparation
        """
        state_preparation = pq.Program()
        for qubit_index in xrange(num_qubits):
            state_preparation.inst(H(qubit_index))
        prepared_term = state_preparation  + gradient_term
        return prepared_term
    return add_state_preparation

def generate_phase_correction(ancilla_qubit_index):
    """
    See Ekert (2002) for the original idea behind this circuit
    :param (int) ancilla_qubit_index: the qubit used as control
    :return (function) add_phase_correction: its like interferometry
    """
    def add_phase_correction(gradient_term):
        """
        Adds the ancilla qubit phase correction operations
        :param (pq.Program) gradient_term: original uncorrected term
        :return (pq.Program) phase_corrected_term: with the ancilla operations
        """
        phase_corrected_term = pq.Program()
        phase_corrected_term.inst(H(ancilla_qubit_index))
        phase_corrected_term += gradient_term
        phase_corrected_term.inst(S(ancilla_qubit_index))
        phase_corrected_term.inst(H(ancilla_qubit_index))
        return phase_corrected_term
    return add_phase_correction

def get_gradient_cost_hamiltonian(cost_hamiltonian, num_qubits):
    """
    Measuring the Ancilla in the Z-basis cancels terms without the desired phase
    :param (PauliSum) cost_hamiltonian: for the expectation value
    :param (int) num_qubits:
    """
    ancilla_qubit_term = PauliTerm("Z", num_qubits)
    full_cost_hamiltonian = cost_hamiltonian*ancilla_qubit_term
    return full_cost_hamiltonian

def generate_get_expectation_value(gradient_cost_hamiltonian, qvm_connection):
    """
    Creates the function which computes the expectation value of a gradient term
    :param (PauliSum) gradient_cost_hamiltonian: the expectation value of this
    :param (api.SyncConnection) qvm_connection: sends programs to the qvm
    """
    def get_expectation_value(gradient_term):
        """
        Computes the expectation value of a given program from the gradient
        :param (pq.Program) gradient_term: one program from the gradient
        :return (float) numerical_expectation_value: should be a real_value
        """
        numerical_expectation_value = expectation_value.expectation(
            gradient_term, gradient_cost_hamiltonian, qvm_connection)
        return numerical_expectation_value
    return get_expectation_value

def compose_programs(programs):
    """
    Compose a list of programs
    :param (list[pq.Program]) programs: compose these
    :return (pq.Program) composed_program: composed
    """
    composed_program = pq.Program()
    for program in programs:
        composed_program += program
    return composed_program

def get_branch_expectation_value(product_expectation_values):
    """
    Get the expectation value of a branch from its constituent products
    :param (list[float]) product_expectation_values: from the product programs
    :reutrn (float) branch_expectation_value: add the product expectation values
    """
    branch_expectation_value = sum(product_expectation_values)
    return branch_expectation_value

def generate_analytical_gradient(hamiltonians, cost_hamiltonian,
        qvm_connection, steps, num_qubits):
    """
    Generates the analytical gradient corresponding to a list of hamiltonians
    :param (list[pq.Program]) hamiltonians: these must be in the correct order
    :param (PauliSum) cost_hamiltonian: for the expectation values
    :param (int) steps: the number of levels in the trotterization
    :param (int) num_qubits: the number of qubits allocated
    :return (function) gradient: expectation value of the analytical_gradient
    """
    p_unitaries = [maxcut_qaoa_core.exponential_map_hamiltonian(hamiltonian)
                   for hamiltonian in hamiltonians]
    def gradient(steps_parameters):
        """
        Computes the expectation values of the gradient on the cost hamiltonian
        :param (list[float]) steps_parameters: for each unitary for each step
        :return (list[float]) branches_expectation_values: gradient entries
        """
        repeated_p_unitaries = p_unitaries*steps
        repeated_hamiltonians = hamiltonians*steps
        assert len(repeated_hamiltonians) == len(steps_parameters)

        make_controlled = generate_make_controlled(num_qubits)
        evaluate_product = generate_evaluate_product(steps_parameters)
        add_phase_correction = generate_phase_correction(num_qubits)
        add_state_preparation = generate_state_preparation(num_qubits)
        gradient_cost_hamiltonian = get_gradient_cost_hamiltonian(
            cost_hamiltonian, num_qubits)
        get_product_expectation_value = generate_get_expectation_value(
            gradient_cost_hamiltonian, qvm_connection)

    	sum_of_branches = differentiate_product_rule(
            repeated_p_unitaries, repeated_hamiltonians, make_controlled)

        evaluated_products = map_products(sum_of_branches, evaluate_product)
        composed_products = map_products(evaluated_products, compose_programs)
        phase_corrected_products = map_products(composed_products,
            add_phase_correction)
        state_prepared_products = map_products(phase_corrected_products,
            add_state_preparation)
        products_expectation_values = map_products(state_prepared_products,
            get_product_expectation_value)
        branches_expectation_values = map_branches(products_expectation_values,
            get_branch_expectation_value)

        return branches_expectation_values
    return gradient
