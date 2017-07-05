"""
This module implements the finite difference gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the gradient for the qaoa algorithm.
"""

import pyquil.quil as pq
from maxcut_qaoa_core import *


def get_finite_difference_component_qaoa(program_parameterizer, params,
        difference_step, component_index):
    #How should I set the size of the difference step?
    lower_difference_params = params[:]
    upper_difference_params = params[:]
    lower_difference_params[component_index] -= difference_step
    upper_difference_params[component_index] += difference_step
    lower_program = program_parameterizer(lower_difference_params)
    upper_program = program_parameterizer(upper_difference_params)
    return lower_program, upper_program

test_structure = {
    "driver":
        {1 : ["a_gate"],
         2 : ["a_gate"]},
    "cost":
        {1 : ["a_gate", "b_gate"],
         2 : ["a_gate", "b_gate"]}
    }

if __name__ == "__main__":
    test_graph_edges = [(0,1)]
    steps = 1
    program_parameterizer, cost_hamiltonian = maxcut_qaoa_constructor(
        test_graph_edges, steps)

    beta = 1.5
    gamma = 1.2
    test_params = [beta, gamma]

    difference_step = 0.1
    component_index = 0 # Will need to incorporate an offset for the state prep

    lower_program, upper_program = get_finite_difference_component_qaoa(
        program_parameterizer, test_params, difference_step, component_index)

    qvm_connection = api.SyncConnection()
    lower_expectation = expectation_value.expectation(lower_program,
        cost_hamiltonian, qvm_connection)
    print(lower_expectation)
    upper_expectation = expectation_value.expectation(upper_program,
        cost_hamiltonian, qvm_connection)
    print(upper_expectation)
    finite_difference = (upper_expectation - lower_expectation)/difference_step
    print(finite_difference)
