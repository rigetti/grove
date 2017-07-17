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
    Maps a program to
    :param (pq.Program) program_branch:
    :param (function) p_unitary:
    :return (function) p_analytical_derivative:
    """
    def p_analytical_derivative(param):
        return program_branch + p_unitary(param)
    return p_analytical_derivative

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

def differentiate_product_rule(p_unitaries_list, hamiltonians_list,
        make_controlled):
    """
    :param (list[function]) p_unitaries_list:
    :param (list[PauliSum]) hamiltonians_list:
    :param (function) make_controlled:
    :return (list[list[function]]) differentiated_product:
    """
    differentiated_product = []
    for unitary_index in xrange(len(p_unitaries_list)):
        p_analytical_derivative_branches = differentiate_unitary(
            p_unitaries_list[unitary_index], hamiltonians_list[unitary_index],
            make_controlled)
        differentiated_term = parallelize(p_analytical_derivative_branches,
                p_unitaries_list, unitary_index)
        differentiated_product.append(differentiated_term)
    return differentiated_product

#The role of the index is confusing!
def differentiate_unitary_in_list(step_index,
        hamiltonian, unitaries_list, make_controlled):
    """
    Computes the analytical derivative of a unitary in a list of unitaries
    :param (int) step_index: the index of selected unitary in the list
    :param (PauliSum) hamiltonian: the hamiltonian which generates the unitaries
    :param (list[pq.Program()]) unitaries_list: each from a term in the PauliSum
    :param (function) make controlled: Maps Pauli operators to controlled gates
    :return (list[list[pq.Program()]]) analytical_derivative_list_branches:
    """
    analytical_derivative_branches = differentiate_unitary(
        unitaries_list[step_index], hamiltonian, make_controlled)
    analytical_derivatives_list_branches = []
    for analytical_derivative_branch in analytical_derivative_branches:
        analytical_derivatives_list_branch = [analytical_derivative_branch
            if idx == step_index else unitary
            for idx, unitary in enumerate(unitaries_list)] #(list[pq.Program()])
        analytical_derivatives_list_branches.append(
            analytical_derivatives_list_branch)
    return analytical_derivatives_list_branches

def zip_analytical_derivatives_list_branches(
        analytical_derivatives_list_branches, unitaries_lists,
        hamiltonian_index):
    """
    Creates a an analytical_derivative program with branches for each term
    :param (list[list[pq.Program()]]) analytical_derivatives_list_branches:
    :param (list[list[pq.Program()]]) unitaries_lists: [ham][branch]
    :param (int) hamiltonian_index: the index of the corresponding hamiltonian
    :return (list[pq.Program()]) analytical_derivative_branches: for each term
    """
    analytical_derivative_branches = []
    for analytical_derivatives_list_branch in analytical_derivatives_list_branches:
        analytical_derivative_branch_slices = []
        for jdx, unitaries_list in enumerate(unitaries_lists):
            if jdx == hamiltonian_index:
                analytical_derivative_branch_slices.append(
                    analytical_derivatives_list_branch)
        else:
            analytical_derivative_branch_slices.append(unitaries_list)
        analytical_derivative_branch = maxcut_qaoa_core.zip_programs_lists(
            analytical_derivative_branch_slices)
        analytical_derivative_branches.append(analytical_derivative_branch)
    return analytical_derivative_branches

def generate_step_analytical_gradient(unitaries_lists, hamiltonians,
        make_controlled):
    """
    Generates a function which finds the derivatives at a given step
    :param (list[list[pq.Program]]) unitaries_lists: [ham][branch]
    :param (list[PauliSum]) hamiltonians: each one generates a unitaries_list
    :param (function) make_controlled: Maps Pauli Operators to controlled gates
    :return (function) step_analytical_gradient: gradient for a given step
    """
    assert len(unitaries_lists) == len(hamiltonians)

    def step_analytical_gradient(step_index):
        """
        Computes all the derivatives at a given step and zips unitaries
        :param (int) step_index: the step to compute the derivatives at
        :return (list[list[pq.Program]]) analytical_gradient: [ham][branch]
        """
        analytical_gradient = []
        for idx, unitaries_list in enumerate(unitaries_lists):
            analytical_derivatives_list_branches = \
                differentiate_unitary_in_list(
                step_index, hamiltonians[idx], unitaries_list, make_controlled)
            analytical_derivative_branches = \
                zip_analytical_derivatives_list_branches(
                analytical_derivatives_list_branches, unitaries_lists, idx)
            analytical_gradient.append(analytical_derivative_branches)
        return analytical_gradient

    return step_analytical_gradient





def extend_cost_hamiltonian(cost_hamiltonian, ancilla_qubit_index):
    ancilla_qubit_term = PauliTerm("Z", ancilla_qubit_index)
    full_cost_hamiltonian = cost_hamiltonian*ancilla_qubit_term
    return full_cost_hamiltonian

#Reduce the complexity!
def insert_ancilla_controlled_hermitian(gradient_component_programs,
        hermitian_structure, hamiltonian_type, ancilla_qubit_index):
    gradient_component_hermitian_operators = hermitian_structure[
        step_index][hamiltonian_type]
    new_gradient_component_programs = []
    for op_type, gate in gradient_component_hermitian_operators:
        operator_name = gate.operator_name
        controlled_operator = make_controlled(operator_name)
        qubit_index = gate.arguments[0]
        controlled_gate = controlled_operator(ancilla_qubit_index, qubit_index)
        if hamiltonian_type == "driver":
            new_gradient_component_programs.append(gradient_component_programs[0] +
                                                   controlled_gate)
        if hamiltonian_type == "cost":
            gradient_component_programs[0].inst(controlled_gate)
            new_gradient_component_programs = gradient_component_programs
    return new_gradient_component_programs


#Reduce the complexity!
def get_analytical_gradient_component_qaoa(params,
        reference_state_program, num_qubits, step_index, hamiltonian_type):
    """
    Computes a single component of the analytical gradient
    """
    gradient_component_programs = [reference_state_program]
    unitary_program, unitary_structure, hermitian_structure = \
        program_parameterizer(params)
    ancilla_qubit_index = num_qubits
    for gradient_component_program in gradient_component_programs:
        gradient_component_program.inst(H(ancilla_qubit_index))

    for free_step_index in unitary_structure:
        step_unitary_structure = unitary_structure[free_step_index]
        for free_hamiltonian_type in step_unitary_structure:

            if (free_step_index == step_index and
                    free_hamiltonian_type == hamiltonian_type):
                gradient_component_programs = insert_ancilla_controlled_hermitian(
                    gradient_component_programs, hermitian_structure,
                    hamiltonian_type, ancilla_qubit_index)

            hamiltonian_operators = step_unitary_structure[free_hamiltonian_type]
            for op_type, gate in hamiltonian_operators:
                for gradient_component_program in gradient_component_programs:
                    gradient_component_program.inst(gate)

    for gradient_component_program in gradient_component_programs:
        gradient_component_program.inst(S(ancilla_qubit_index))
        gradient_component_program.inst(H(ancilla_qubit_index))
    return gradient_component_programs
