"""
This module implements the analytical gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the gradient of a given product of parametric unitary operators.
"""

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

def construct_ref_state_prep(num_qubits):
    """Constructs the standard reference state for QAOA - "s"
    """
    ref_prog = pq.Program()
    for i in xrange(num_qubits):
        ref_prog.inst(H(i))
    ref_state_prep = ref_prog
    return ref_state_prep

def define_parametric_program_qaoa(trotterization_steps, graph):
    cost_para_programs = []
    driver_para_programs = []

    for step in xrange(trotterization_steps):
        for qubit_index_A, qubit_index_B in graph.edges():
            cost_para_program = []
            interaction_term = (PauliTerm("Z", qubit_index_A, 0.5)*
                                PauliTerm("Z", qubit_index_B))
            constant_term = PauliTerm("I", 0, -0.5)
            cost_para_program.append(exponential_map(interaction_term))
            cost_para_program.append(exponential_map(constant_term))

        for qubit_index in graph.nodes():
            driver_para_program = []
            driver_term = PauliTerm("X", qubit_index, -1.0)
            driver_para_program.append(exponential_map(driver_term))

        cost_para_programs.append(cost_para_program)
        driver_para_programs.append(driver_para_program)

    return (cost_para_programs, driver_para_programs)


def get_parameterized_program_qaoa(steps, cost_para_programs,
        driver_para_programs, ref_state_prep):
    """
    Return a function that accepts parameters and returns a new Quil
    program

    :returns: a function
    """

    def parameterize_prog(params):
	"""Construct a Quil program for the vector (beta, gamma).

	:param params: array of 2*p angles, betas first, then gammas
	:return: a pyquil program object
	"""
	if len(params) != 2*steps:
	    raise ValueError("""params doesn't match the number of parameters set
				by `steps`""")
	betas = params[:steps]
	gammas = params[steps:]

	parameterized_prog = pq.Program()
	parameterized_prog += ref_state_prep
	for step_index in xrange(steps):
            cost_para_program = cost_para_programs[step_index]
            step_gammas = gammas[step_index]
	    for cost_term in cost_para_program: #This is longer than driver
                step_cost_term = cost_term(step_gammas)
		parameterized_prog += step_cost_term
            cost_length = len(cost_para_program)

            driver_para_program = driver_para_programs[step_index]
            step_betas = betas[step_index]
	    for driver_term in driver_para_program:
                step_driver_term = driver_term(step_betas)
		parameterized_prog += step_driver_term

	return parameterized_prog

    return parameterize_prog


def get_analytical_gradient_component_qaoa(params, parameterize_prog, graph_size,
    cost_length, component_index):
    """
    Maps: Parameterized_Program -> Parameterized_Program

    :returns: a function
    """
    gradient_prog = pq.Program()
    parameterized_prog = parameterize_prog(params)
    parameterized_prog_lower = parameterized_prog[:component_index-1]
    parameterized_prog_upper = parameterized_prog[component_index:]
    ancilla_qubit_index = graph_size #Check how to add a new qubit
    prog += parameterized_prog_lower
    prog.inst(H(ancilla_qubit_index))
    if component_index > cost_length: #Then apply driver gate
        node_index = component_index_to_graph_element(component_index)
        prog.inst(CNOT(node_index, ancilla_qubit_index)) #Check the order on this!
    else:
        edge_index_A, edge_index_B = component_index_to_graph_element(component_index)
        prog.inst(CZ(edge_index_A, ancilla_qubit_index))
        prog.inst(CZ(edge_index_B, ancilla_qubit_index))
    prog += parameterized_prog_upper
    i_one = np.array([[1.0, 0.0], [0.0, 1.0j]])
    prog.defgate("I-ONE", i_one)
    prog.inst("I-ONE", ancilla_qubit_index)
    prog.inst(H(ancilla_qubit_index))
    return prog

if __name__ == "__main__":
    square_ring_edges = [(0,1), (1,2), (2,3), (3,0)]
    graph = edges_to_graph(square_ring_edges)
    trotterization_steps = 2 #Referred to as "p" in the paper
    cost_para_programs, driver_para_programs = define_parametric_program_qaoa(
        trotterization_steps, graph)
    #cost_ham = get_cost_ham(graph)
    #ref_ham = get_ref_ham(graph)


