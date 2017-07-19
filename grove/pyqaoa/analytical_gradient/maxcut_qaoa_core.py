"""
This module prepares and evaluates the MAXCUT QAOA expectation value.
"""

import numpy as np
import networkx as nx
import pyquil.quil as pq
from pyquil.gates import *
from pyquil.paulis import *

def edges_to_graph(graph_edges):
    """
    Converts a list of edges into a networkx graph object
    :param (list[tuple[int]]) graph_edges: list of int pairs e.g. [(0,1), (0,2)]
    :return (networkx.Graph) graph: a graph defined by the given edges_list
    """
    maxcut_graph = nx.Graph()
    for edge in graph_edges:
        maxcut_graph.add_edge(*edge)
    graph = maxcut_graph.copy()
    return graph

def construct_reference_state_program(num_qubits):
    """
    Constructs the program for the standard reference state for QAOA - "s"
    :param (int) num_qubits: the number of qubits in the reference state
    :return (Program) reference_state_program: the pyquil program
    """
    reference_state_program = pq.Program()
    for qubit_index in xrange(num_qubits):
        gate = H(qubit_index)
        reference_state_program.inst(gate)
    return reference_state_program

def get_cost_hamiltonian(graph):
    """
    Constructs the cost hamiltonian for a given graph
    :param (networkx.Graph) graph: a graph where each node is a qubit index
    :return (PauliSum) cost_hamiltonian: the hamiltonian for the cost function
    :return (pq.Program) cost_program: evolves a state with the
    """
    cost_hamiltonian = PauliSum([])
    for qubit_index_A, qubit_index_B in graph.edges():
        cost_term = PauliSum([PauliTerm("Z", qubit_index_A, 1.0)*
                              PauliTerm("Z", qubit_index_B)])
        cost_hamiltonian += cost_term
    return cost_hamiltonian

def get_driver_hamiltonian(graph):
    """
    Constructs the driver hamiltonian for a given graph
    :param (networkx.Graph) graph: a graph where each node is a qubit index
    :return (PauliSum) driver_hamiltonian: hamiltonian for the driver function
    :return (pq.Program) driver_program: evolves a state according to the driver
    """
    driver_hamiltonian = PauliSum([])
    for qubit_index in graph.nodes():
        driver_term = PauliSum([PauliTerm("X", qubit_index, 1.0)])
        driver_hamiltonian += driver_term
    return driver_hamiltonian

def exponential_map_hamiltonian(hamiltonian):
    """
    Generates a unitary operator from a hamiltonian
    :param (PauliSum) hamiltonian: a hermitian hamiltonian
    :param (float) parameter: the coefficient in the exponential
    :return (function) p_unitary: the generated parameterized unitary
    """
    p_unitary_list = []
    for term in hamiltonian:
        p_unitary_term = exponential_map(term)
        p_unitary_list.append(p_unitary_term)

    def p_unitary(param):
        p_unitary_program = pq.Program()
        for p_unitary_term in p_unitary_list:
            p_unitary_program += p_unitary_term(param)
        return p_unitary_program
    return p_unitary

def exponentiate_hamiltonian(hamiltonian, parameter):
    """
    Generates a unitary operator from a hamiltonian and a parameter
    :param (PauliSum) hamiltonian: a hermitian hamiltonian
    :param (float) parameter: the coefficient in the exponential
    :return (pq.Program) unitary: the program generated from the hamiltonian
    """
    unitary = pq.Program()
    for term in hamiltonian:
        exponentiated_term = exponential_map(term)
        unitary_term = exponentiated_term(parameter)
        unitary += unitary_term
    return unitary

def get_program_parameterizer(steps, hamiltonian):
    """
    Creates a function for parametrically generating maxcut qaoa pyquil programs
    :param (int) steps: also known as the "level" or "p"
    :param (PauliSum) hamiltonian: generates the parameterized unitaries
    :return (function) program_parameterizer: maps angles to a pyquil program
    """

    def program_parameterizer(angles):
        """
        Creates a list of parameterized unitaries
        :param (list[floats]) angles: the exponentiation angle for each step
        :return (list[pq.Program()]) unitaries list: generated from hamiltonian
        """
        if len(angles) != steps:
            raise ValueError("""len(params) doesn't match the number of
                parameters required by `steps`""")
        unitaries_list = [exponentiate_hamiltonian(hamiltonian, step_angle/2)
            for step_angle in angles]
        return unitaries_list

    return program_parameterizer

def zip_programs_lists(programs_lists):
    """
    Creates a program by fusing programs from a list
    :param (list[list[pq.Program]]) programs_lists:
    :param (pq.Program) fused_program:
    """
    programs_lengths = [len(programs_list) for programs_list in programs_lists]
    assert programs_lengths[1:] == programs_lengths[:-1]
    zipped_programs = zip(*programs_lists)
    fused_program = pq.Program()
    for programs_slice in zipped_programs:
        for program in programs_slice:
            fused_program += program
    return fused_program
