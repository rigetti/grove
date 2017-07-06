"""
This module implements the analytical gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the analytical gradient for the qaoa algorithm.
"""


import numpy as np
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *

import maxcut_qaoa_core
import expectation_value


def get_analytical_gradient_component_qaoa(program_parameterizer, params,
        num_qubits, step_index, hamiltonian_type):
    """
    Computes a single component of the analytical gradient
    """
    gradient_component_program = pq.Program()
    unitary_program, unitary_structure, hermitian_structure = \
        program_parameterizer(params)

    ancilla_qubit_index = num_qubits
    gradient_component_program.inst(H(ancilla_qubit_index))

    for free_step_index in unitary_structure:
        step_unitary_structure = unitary_structure[free_step_index]
        for free_hamiltonian_type in step_unitary_structure:

            if (free_step_index == step_index and
                    free_hamiltonian_type == hamiltonian_type):
                gradient_component_hermitian_operators = hermitian_structure[
                    step_index][hamiltonian_type]
                for op_type, gate in gradient_component_hermitian_operators:
                    gradient_component_program += gate

            hamiltonian_operators = step_unitary_structure[free_hamiltonian_type]
            for op_type, gate in hamiltonian_operators:
                gradient_component_program += gate

    gradient_component_program.inst(S(ancilla_qubit_index))
    gradient_component_program.inst(H(ancilla_qubit_index))

    return gradient_component_program


if __name__ == "__main__":
    test_graph_edges = [(0,1)]
    steps = 1
    program_parameterizer, reference_state_program, cost_hamiltonian, num_qubits = \
        maxcut_qaoa_core.maxcut_qaoa_constructor(test_graph_edges, steps)

    beta = np.pi/8
    gamma = np.pi/4
    #beta = 1.5
    #gamma = 1.2
    params = [beta, gamma]

    step_index = 1
    hamiltonian_type = "cost"

    gradient_component_program = get_analytical_gradient_component_qaoa(
        program_parameterizer, params, num_qubits, step_index, hamiltonian_type)
    #print(gradient_component_program)
    full_program = reference_state_program + gradient_component_program
    print(full_program)
    qvm_connection = api.SyncConnection()
    numeric_expectation = expectation_value.expectation(full_program,
        cost_hamiltonian, qvm_connection)
    print(numeric_expectation)
