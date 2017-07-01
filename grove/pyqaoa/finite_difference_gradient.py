"""
This module implements the finite difference gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the gradient for the qaoa algorithm.
"""

import networkx as nx
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import *


def edges_to_graphs(edges_list):
    """Converts a list of edges into a networkx graph object
    """
    maxcut_graph = nx.Graph()
    for edge in edges_list:
        maxcut_graph.add_edge(*edge)
    graph = maxcut_graph.copy()
    return graph


def construct_ref_state_prep(num_qubits):
    """Constructs the standard reference state for QAOA - "s"
    """
    ref_prog = pq.Program()
    for qubit_index in xrange(num_qubits):
        ref_prog.inst(H(qubit_index)
    ref_state_prep = ref_prog
    return ref_state_prep


def define_qaoa_programs(trotterization_steps, graph):
    cost_programs = []
    driver_programs = []

    for step in xrange(trotterization_steps):
        for qubit_index_A, qubit_index_B in graph.edges():
            cost_program = []
            interaction_term = (PauliTerm("Z", qubit_index_A)* #Ignore 0.5
                                PauliTerm("Z", qubit_index_B))
            #constant_term = PauliTerm("I", 0, -0.5) #Ignore constant
            cost_program.append(interaction_term)
            #cost_program.append(constant_term) #Ignore constant

        for qubit_index in graph.nodes():
            driver_program = []
            driver_term = PauliTerm("X", qubit_index, -1.0)
            driver_program.append(driver_term)

        cost_programs.append(cost_program)
        driver_programs.append(driver_program)

    return (cost_programs, driver_programs)


def get_program_parameterizer_qaoa(steps, cost_programs, driver_programs):
    """
    Return a function that accepts parameters and returns a list of
    pyquil programs constituting a full pyquil program
    """

    def program_parameterizer(params):
        """Constructs a list of pyquil programs constituting the full pyquil
            program for the vector (beta, gamma).
        :param params: an array of 2*p angles, betas first, then gammas
        :return: a list of pyquil programs
        """
        if len(params) != 2*steps:
            raise ValueError("""len(params) doesn't match the number of
                parameters required by `steps`"""

        betas = params[:steps]
        gammas = params[steps:]

        terms_list = []

        for step_index = xrange(steps):
            cost_program = cost_programs[step_index]
            step_gammas = gammas[step_index]
            for cost_term in cost_program:
                parameterized_cost_term = exponential_map(cost_term)
                step_cost_term = parameterized_cost_term(step_gammas)
                terms_list.append(step_cost_term)

            driver_program = driver_programs[step_index]
            step_betas = betas[step_index]
            for driver_term in driver_program:
                parameterized_driver_term = exponential_map(driver_term)
                step_driver_term = parameterized_driver_term(step_betas)
                terms_list.append(step_driver_term)

            return terms_list

        return program_parameterizer


def get_finite_difference_component_qaoa(program_parameterizer, params,
        difference_step, num_qubits, component_index):
    #How do I set the size of the difference step?
    lower_difference_prog = pq.Program()
    upper_difference_prog = pq.Program()
    lower_difference_params = params[:]
    lower_difference_params[component_index] -= difference_step
    upper_difference_params = params[:]
    upper_difference_params[component_index] += difference_step
    lower_terms_list = program_parameterizer(lower_difference_params)
    upper_terms_list = program_parameterizer(upper_difference_params)


def expectation(pyquil_program, pauli_sum, qvm):
    """
    Computes the expectation value of the pauli_sum over the distribution
    generated from pyquil_prog.

    :param pyquil_program: (pyquil program)
    :param pauli_sum: (PauliSum) PauliSum representing the operator of which
                      to calculate the expectation value.
    :param qvm: (qvm connection)

    :returns: (float) representing the expectation value of pauli_sum given
              the distribution generated from quil_prog.
    """
    operator_programs = []
    operator_coefficients = []
    for pauli_term in pauli_sum.terms:
        operator_program = pq.Program()
        for qubit_index, operator in p_term:
            operator_program.inst(STANDARD_GATES[op](qindex))
        operator_programs.append(operator_program)
        operator_coefficients.append(pauli_term.coefficient)

    result_overlaps = qvm.expectation(pyquil_program,
        operator_programs=operator_programs)
    result_overlaps = list(result_overlaps)
    assert len(result_overlaps) == len(operator_programs), """Incorrect number
        of results were returned from the QVM"""
    expectation = sum(map(lambda x: x[0]*x[1], zip(result_overlaps, operator_coeffs)))
    return expectaion.real


if __name__ == "__main__":



