"""
This module implements the analytical gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the analytical gradient for the qaoa algorithm.
"""


import numpy as np
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import *

import maxcut_qaoa_core
import expectation_value

#Need to generalize this to make arbitrary one qubit ops controlled
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
    :param (list[function]) p_unitaries_product: [uni] i.e. unitary_index
    :param (list[PauliSum]) hamiltonians_list: for each factor in the product
    :param (function) make_controlled: e.g. X -> CNOT
    :return (list[list[function]]) differentiated_product: [summand*branch][uni]
    """
    sum_of_differentiated_products = []
    for unitary_index in xrange(len(p_unitaries_product)):
        p_analytical_derivative_branches = differentiate_unitary(
            p_unitaries_product[unitary_index],
            hamiltonians_list[unitary_index], make_controlled)
        summand = parallelize(p_analytical_derivative_branches,
                p_unitaries_product, unitary_index)
        sum_of_differentiated_products += summand
    return sum_of_differentiated_products

def generate_evaluate_product(parameters):
    """
    Creates the function which evaluates a product of parameterized operators
    """
    def evaluate_product(product):
        assert len(product) == len(parameters)
        evaluated_product = []
        for factor, parameter in zip(product, parameters):
            evaluated_factor = factor(parameter)
            evaluated_product.append(evaluated_factor)
        return evaluated_product
    return evaluate_product

def map_products(sum_of_products, product_map):
    """
    Map each product in the sum of products using the given map
    :param (list[list[Abstract_1]]) sum_of_products:
    :param (function: list[Abstract_1] -> Abstract_2) product_map:
    :return (list[Abstract_2]) mapped_sum:
    """
    mapped_sum = []
    for product in sum_of_products:
        mapped_product = product_map(product)
        mapped_sum.append(mapped_product)
    return mapped_sum

def map_factors(sum_of_products, factor_map):
    """
    Map each factor in each of the products using the given map
    :param (list[list[Abstract_1]]) sum_of_products:
    :param (function: Abstract_1 -> Abstract_2) factor_map:
    :return (list[list[Abstract_2]) mapped_sum:
    """
    mapped_sum = []
    for product in sum_of_products:
        mapped_product = []
        for factor in product:
            mapped_factor = factor_map(factor)
            mapped_product.append(mapped_factor)
        mapped_sum.append(mapped_product)
    return mapped_sum

def generate_state_preparation(num_qubits):
    """
    Generates function for prepending state preparation gates
    :param (int) num_qubits:
    :return (function) add_state_preparation:
    """
    def add_state_preparation(gradient_term):
        """
        Prepends state preparation gates to a given term in the gradient
        """
        state_preparation = pq.Program()
        for qubit_index in xrange(num_qubits):
            state_preparation.inst(H(qubit_index))
        prepared_term = state_preparation  + gradient_term
        return prepared_term
    return add_state_preparation

def generate_phase_correction(ancilla_qubit_index):
    def add_phase_correction(gradient_term):
        """
        Adds the ancilla qubit phase correction operations
        :param (pq.Program) gradient_term: original uncorrected term
        :param (int) ancilla_qubit_index: the qubit used as control
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
    Extends the cost hamiltonian
    """
    ancilla_qubit_term = PauliTerm("Z", num_qubits)
    full_cost_hamiltonian = cost_hamiltonian*ancilla_qubit_term
    return full_cost_hamiltonian

def generate_expectation_value(gradient_cost_hamiltonian, qvm_connection):
    def get_expectation_value(gradient_term):
        numerical_expectation_value = expectation_value.expectation(
            gradient_term, gradient_cost_hamiltonian, qvm_connection)
        return numerical_expectation_value
    return get_expectation_value

def compose_programs(programs):
    composed_program = pq.Program()
    for program in programs:
        composed_program += program
    return composed_program

def generate_analytical_gradient(hamiltonians, cost_hamiltonian,
        qvm_connection, steps, num_qubits):
    """
    Generates the analytical gradient corresponding to a list of hamiltonians
    :param (list[pq.Program]) hamiltonians:
    :param (function) make_controlled:
    :param (int) steps:
    :return ? :
    """
    p_unitaries = [maxcut_qaoa_core.exponential_map_hamiltonian(hamiltonian)
                   for hamiltonian in hamiltonians]
    def gradient(steps_parameters):
        repeated_p_unitaries = p_unitaries*steps
        repeated_hamiltonians = hamiltonians*steps
        assert len(repeated_hamiltonians) == len(steps_parameters)

        make_controlled = generate_make_controlled(num_qubits)
        evaluate_product = generate_evaluate_product(steps_parameters)
        add_phase_correction = generate_phase_correction(num_qubits)
        add_state_preparation = generate_state_preparation(num_qubits)
        gradient_cost_hamiltonian = get_gradient_cost_hamiltonian(
            cost_hamiltonian, num_qubits)
        get_expectation_value = generate_expectation_value(
            gradient_cost_hamiltonian, qvm_connection)

    	sum_of_products = differentiate_product_rule(
            repeated_p_unitaries, repeated_hamiltonians, make_controlled)

        evaluated_products = map_products(sum_of_products, evaluate_product)
        def print_func(x):
            print(x)
        #map_factors(evaluated_products, print_func)
        composed_products = map_products(evaluated_products, compose_programs)
        #map_products(composed_products, print_func)
        phase_corrected_products = map_products(composed_products,
            add_phase_correction)
        #map_products(phase_corrected_products, print_func)
        state_prepared_products = map_products(phase_corrected_products,
            add_state_preparation)
        map_products(state_prepared_products, print_func)
        gradient_values = map_products(state_prepared_products,
            get_expectation_value)

        return gradient_values
    return gradient
