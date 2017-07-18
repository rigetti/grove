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

def parallelize(column, partial_row, new_column_index):
    """
    Map a column and a row to a full matrix
    :param (list[Abstract]) column: the new elements to be added
    :param (list[Abstract]) partial_row: replace elements at new_column_index
    :param (int) new_column_index: the index in partial row to be replaced
    :return (list[list[Abstract]]) matrix: the generated matrix
    """
    matrix = []
    for column_element in column:
        row = [column_element if column_index == new_column_index
               else row_element for column_index, row_element
               in enumerate(partial_row)]
        matrix.append(row)
    return matrix

#Currently Debugging Here!

#Flatten this final structure
#All the sums can be over the same list
def differentiate_product_rule(p_unitaries_product, hamiltonians_list,
        make_controlled):
    """
    :param (list[function]) p_unitaries_product: [ham]
    :param (list[PauliSum]) hamiltonians_list: for each factor in the product
    :param (function) make_controlled: e.g. X -> CNOT
    :return (list[list[list[function]]]) differentiated_product: indexing below
    """
    #gain one index for branches of the derivative
    #gain one index for product rule
    #Indexing should be [summand*branch][ham]
    #i.e. just list[list[function]]?
    differentiated_product = [] #[summand][ham][branch]
    for unitary_index in xrange(len(p_unitaries_product)):
        #(list[function])
        p_analytical_derivative_branches = differentiate_unitary(
            p_unitaries_product[unitary_index],
            hamiltonians_list[unitary_index], make_controlled)
        #(list[list[function]])
        derivative_summand = parallelize(p_analytical_derivative_branches,
                p_unitaries_product, unitary_index)
        differentiated_product.append(derivative_summand)
    return differentiated_product

#Flatten this structure
def evaluate_differentiated_product(differentiated_product, parameters,
        program_maps=[]):
    """
    Evaluates each term in the differentiated product using the given parameters
    :param (list[list[list[function]]]) differentiated_product:
    :param (list[float]) parameters:
    """
    evaluated_product = []
    for summand in differentiated_product:
        evaluated_summand = []
        for factor in summand:
            evaluated_factor = []
            for branch, param in zip(factor, parameters):
                evaluated_branch = branch(param)
                for program_map in program_maps:
                    evaluated_branch = program_map(evaluated_branch)
                evaluated_factor.append(evaluated_branch)
            evaluated_summand.append(evaluated_factor)
        evaluated_product.append(evaluated_summand)
    return evaluated_product

def generate_state_preparation(num_qubits):
    def add_state_preparation(gradient_term):
        state_preparation = pq.Program()
        for qubit_index in num_qubits:
            state_preparation.inst(H(qubit_index))
        prepared_term = state_preparation  + gradient_term
        return prepared_term
    return add_state_preparation

#Based on the 2002 Ekert Paper
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

#need to simplify the return type
def generate_analytical_gradient(hamiltonians, steps, num_qubits):
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
        add_state_preparation = generate_state_preperation(num_qubits)
        add_phase_correction = generate_phase_correction(num_qubits)

    	differentiated_product = differentiate_product_rule(repeated_p_unitaries,
            repeated_hamiltonians, make_controlled)
        evaluated_gradient = evaluate_differentiated_product(
            differentiated_product, steps_parameters,
            [add_phase_correction, add_state_preparation])
        return evaluated_gradient
    return gradient

def extend_cost_hamiltonian(cost_hamiltonian, ancilla_qubit_index):
    """
    """
    ancilla_qubit_term = PauliTerm("Z", ancilla_qubit_index)
    full_cost_hamiltonian = cost_hamiltonian*ancilla_qubit_term
    return full_cost_hamiltonian

def compute_gradient_term_expectation_value(cost_hamiltonian, gradient_term,
        qvm_connection):
    numerical_expectation_value = expectation_value.expectation(gradient_term,
        cost_hamiltonian, qvm_connection)
    return numerical_expectation_value
