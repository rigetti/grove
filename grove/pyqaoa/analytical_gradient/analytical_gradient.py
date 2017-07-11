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

qvm_connection = api.SyncConnection()


def make_controlled(operator_name):
    mapping_dict = {
        "X": CNOT,
        "Z": CPHASE(np.pi)
        }
    return mapping_dict[operator_name]

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

if __name__ == "__main__":
    test_graph_edges = [(0,1)]
    steps = 1
    program_parameterizer, reference_state_program, cost_hamiltonian, num_qubits = \
        maxcut_qaoa_core.maxcut_qaoa_constructor(test_graph_edges, steps)

    #beta = np.pi/8
    #gamma = np.pi/2
    #beta = np.pi/4
    #gamma = np.pi/4
    beta = 1.3
    gamma = 1.2
    params = [beta, gamma]

    step_index = 1
    hamiltonian_type = "cost"
    #hamiltonian_type = "driver"

    gradient_component_programs = get_analytical_gradient_component_qaoa(
        program_parameterizer, params, reference_state_program,
        num_qubits, step_index, hamiltonian_type)
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
