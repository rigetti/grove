"""
This module implements the finite difference gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the gradient for the qaoa algorithm.
"""

import numpy as np
import networkx as nx
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import *


def edges_to_graph(edges_list):
    """Converts a list of edges into a networkx graph object
    """
    maxcut_graph = nx.Graph()
    for edge in edges_list:
        maxcut_graph.add_edge(*edge)
    graph = maxcut_graph.copy()
    return graph


def construct_reference_state_program(num_qubits):
    """Constructs the standard reference state for QAOA - "s"
    """
    reference_state_program = pq.Program()
    for qubit_index in xrange(num_qubits):
        gate = H(qubit_index)
        reference_state_program.inst(gate)
    return reference_state_program


def define_cost_hamiltonian(graph):
    cost_hamiltonian = PauliSum([])
    for qubit_index_A, qubit_index_B in graph.edges():
        cost_term = PauliSum([PauliTerm("Z", qubit_index_A, 1.0)*
                              PauliTerm("Z", qubit_index_B)])
        cost_hamiltonian += cost_term
    return cost_hamiltonian


def define_driver_hamiltonian(graph):
    driver_hamiltonian = PauliSum([])
    for qubit_index in graph.nodes():
        driver_term = PauliSum([PauliTerm("X", qubit_index, -1.0)])
        driver_hamiltonian += driver_term
    return driver_hamiltonian


def get_program_parameterizer_maxcut(steps, reference_state_program,
        cost_hamiltonian, driver_hamiltonian):
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
                parameters required by `steps`""")

        betas = params[:steps]
        gammas = params[steps:]

        parameterized_program = pq.Program()
        parameterized_program += reference_state_program

        for step_index in xrange(steps):
            this_step_gammas = gammas[step_index]
            #The terms for a given_step are all sigma_z operators and hence commute,
            #so you can simply exponentiate in sequence.
            for cost_hermitian_term in cost_hamiltonian:
                parameterized_cost_unitary_term = exponential_map(
                    cost_hermitian_term)
                this_step_cost_unitary_term = parameterized_cost_unitary_term(
                    this_step_gammas)
                parameterized_program += this_step_cost_unitary_term

            this_step_betas = betas[step_index]
            #The terms for a given step are all sigma_x operators and hence commute,
            #so you can simply exponentiate in sequence.
            for driver_hermitian_term in driver_hamiltonian:
                parameterized_driver_unitary_term = exponential_map(
                    driver_hermitian_term)
                this_step_driver_unitary_term = parameterized_driver_unitary_term(
                    this_step_gammas)
                parameterized_program += this_step_driver_unitary_term

        return parameterized_program

    return program_parameterizer


def expectation(pyquil_program, cost_hamiltonian, qvm_connection):
    """
    Computes the expectation value of the pauli_sum over the distribution
    generated from pyquil_prog.

    :param pyquil_program: (pyquil program)
    :param pauli_sum: (PauliSum) PauliSum representing the operator of which
                      to calculate the expectation value.
    :param qvm_connection: (qvm connection)

    :returns: (float) representing the expectation value of pauli_sum given
              the distribution generated from quil_prog.
    """
    operator_programs = []
    operator_coefficients = []
    for pauli_term in pauli_sum.terms:
        operator_program = pq.Program()
        for qubit_index, operator in pauli_term:
            operator_program.inst(STANDARD_GATES[operator](qubit_index))
        operator_programs.append(operator_program)
        operator_coefficients.append(pauli_term.coefficient)

    result_overlaps = qvm_connection.expectation(pyquil_program,
        operator_programs=operator_programs)
    result_overlaps = list(result_overlaps)
    assert len(result_overlaps) == len(operator_programs), """The incorrect
        number of results were returned from the QVM"""
    expectation = sum(map(lambda x: x[0]*x[1],
                          zip(result_overlaps, operator_coefficients)))
    return expectation.real


def analytic_expectation(beta, gamma): #Mathematica this!
    return np.sin(4*beta)*np.sin(2*gamma)


if __name__ == "__main__":
    test_graph_edges = [(0,1)]
    test_graph = edges_to_graph(test_graph_edges)
    steps = 1 #the number of trotterization steps
    num_qubits = len(test_graph.nodes())
    reference_state_program = construct_reference_state_program(num_qubits)
    cost_hamiltonian = define_cost_hamiltonian(test_graph)
    driver_hamiltonian = define_driver_hamiltonian(test_graph)

    program_parameterizer = get_program_parameterizer_maxcut(steps,
        reference_state_program, cost_hamiltonian, driver_hamiltonian)
    #beta = np.pi
    #gamma = np.pi
    beta = 1.2
    gamma = 1.4
    test_params = [beta, gamma]
    test_program = program_parameterizer(test_params)
    qvm_connection = api.SyncConnection()
    print(analytic_expectation(beta, gamma))
    print(expectation(test_program, cost_hamiltonian, qvm_connection))



