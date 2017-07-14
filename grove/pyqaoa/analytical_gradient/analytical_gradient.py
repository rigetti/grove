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

def hamiltonian_to_programs(hamiltonian, pauli_operator_mapping):
    """
    Converts a hamiltonian to the corresponding list of programs
    :param (PauliSum) hamiltonian: has a list of pauli terms
    :return (list[pq.Program]) programs: the programs from the hamiltonian
    """
    programs = [pauli_term_to_program(pauli_term, pauli_operator_mapping)
        for pauli_term in hamiltonian]
    return programs

def get_analytical_derivative_unitary(unitary, generator,
        pauli_operator_mapping):
    """
    Writes the analytical derivative of a unitary
    :param (pq.Program) unitary: the unitary generated by generator
    :param (PauliSum) generator: to be converted into list[pq.Program]
    :return (list[pq.Program]) analytical_derivative: the derivative programs
    """
    programs = hamiltonian_to_programs(generator, pauli_operator_mapping)
    analytical_derivative = [program + unitary for program in programs]
    return analytical_derivative



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
def get_analytical_gradient_component_qaoa(program_parameterizer, params,
        reference_state_program, num_qubits, step_index, hamiltonian_type):
    """
    Computes a single component of the analytical gradient
    """
    gradient_component_programs = [reference_state_program]
    unitary_program, unitary_structure, hermitian_structure = \
        program_parameterizer(params)
    #The Hermitian Structure needs to flag if the operators are serial or parallel
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

def extend_cost_hamiltonian(cost_hamiltonian, ancilla_qubit_index):
    ancilla_qubit_term = PauliTerm("Z", ancilla_qubit_index)
    full_cost_hamiltonian = cost_hamiltonian*ancilla_qubit_term
    return full_cost_hamiltonian
