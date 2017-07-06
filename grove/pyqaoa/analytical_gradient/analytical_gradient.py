"""
This module implements the analytical gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the analytical gradient for the qaoa algorithm.
"""


import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *

import maxcut_qaoa_core
import expectation_value

def get_analytical_gradient_component_qaoa(program_parameterizer, params,
        num_qubits, step_index, hamiltonian_type):
    """
    Computes a single component of the analytical gradient

    MAKE SURE STEP_INDEX IS CONSISTENT!
    """
    gradient_component_program = pq.Program()
    unitary_program, unitary_structure, hermitian_structure = \
        program_parameterizer(params)

    ancilla_qubit_index = num_qubits
    gradient_component_program.inst(H(ancilla_qubit_index))

    unitary_gates_before_gradient = [unitary_structure[free_step_index] for
        free_step_index in range(1, step_index + 1)]
    for step_gates in unitary_gates_before_gradient:
        for hamiltonian_operators in step_gates.values():
            for op_type, gate in hamiltonian_operators:
                gradient_component_program += gate

    gradient_component_hermitian_gates = hermitian_structure[step_index][hamiltonian_type]
    for op_type, gate in gradient_component_hermitian_gates:
        gradient_component_program += gate

    highest_step = max(unitary_structure.keys())
    unitary_gates_after_gradient = [unitary_structure[free_step_index] for
        free_step_index in range(step_index + 1, highest_step)]
    for step_gates in unitary_gates_after_gradient:
        for hamiltonian_type_gates in step_gates:
            for gate in hamiltonian_type_gates:
                gradient_component_program += gate

    gradient_component_program.inst(S(ancilla_qubit_index))
    gradient_component_program.inst(H(ancilla_qubit_index))
    return gradient_component_program


if __name__ == "__main__":
    test_graph_edges = [(0,1)]
    steps = 1
    program_parameterizer, reference_state_program, cost_hamiltonian, num_qubits = \
        maxcut_qaoa_core.maxcut_qaoa_constructor(test_graph_edges, steps)

    beta = 1.5
    gamma = 1.2
    params = [beta, gamma]

    step_index = 1
    hamiltonian_type = "cost"

    gradient_component_program = get_analytical_gradient_component_qaoa(
        program_parameterizer, params, num_qubits, step_index, hamiltonian_type)
    #print(gradient_component_program)
    full_program = reference_state_program + gradient_component_program
    #print(full_program)
    qvm_connection = api.SyncConnection()
    numeric_expectation = expectation_value.expectation(full_program,
        cost_hamiltonian, qvm_connection)
    print(numeric_expectation)
