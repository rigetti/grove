"""
This module implements the analytical gradient as defined in the paper:
'Practical Optimization for hybrid quantum-classical algorithms'
Compute the gradient of a given product of parametric unitary operators.
"""

import networkx as nx
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import PauliTerm, PauliSum

def edges_to_graph(edges_list):
    """Converts a list of edges into a networkx graph object
    """
    maxcut_graph = nx.Graph()
    for edge in edges_list:
        maxcut_graph.add_edge(*edge)
    graph = maxcut_graph.copy()
    return graph

def get_cost_ham(graph):
    """Constructs the cost Hamiltonian - "C"
    """
    cost_ham = []
    for qubit_index_A, qubit_index_B in graph.edges():
        cost_ham.append((PauliTerm("Z", qubit_index_A, 0.5)*
                               PauliTerm("Z", qubit_index_B)) +
                              PauliTerm("I", 0, -0.5))
    return cost_ham

def get_ref_ham(graph):
    """Constructs the driving Hamiltonian - "B"
    """
    ref_ham = []
    for qubit_index in graph.nodes():
        ref_ham.append(PauliSum([PauliTerm("X", qubit_index, -1.0)]))
    return ref_ham

def construct_ref_state_prep(num_qubits):
    """Constructs the standard reference state for QAOA - "s"
    """
    ref_prog = pq.Program()
    for i in xrange(num_qubits):
        ref_prog.inst(H(i))
    ref_state_prep = ref_prog
    return ref_state_prep

def get_parameterized_program(steps, cost_ham, ref_ham, ref_state_prep):
    """
    Return a function that accepts parameters and returns a new Quil
    program

    :returns: a function
    """
    cost_para_programs = []
    driver_para_programs = []

    for idx in xrange(steps):
	cost_list = []
	driver_list = []
	for cost_pauli_sum in cost_ham:
	    for term in cost_pauli_sum.terms:
		cost_list.append(exponential_map(term))

	for driver_pauli_sum in ref_ham:
	    for term in driver_pauli_sum.terms:
		driver_list.append(exponential_map(term))

	cost_para_programs.append(cost_list)
	driver_para_programs.append(driver_list)

    def psi_ref(params):
	"""Construct a Quil program for the vector (beta, gamma).

	:param params: array of 2 . p angles, betas first, then gammas
	:return: a pyquil program object
	"""
	if len(params) != 2*steps:
	    raise ValueError("""params doesn't match the number of parameters set
				by `steps`""")
	betas = params[:steps]
	gammas = params[steps:]

	prog = pq.Program()
	prog += ref_state_prep
	for idx in xrange(steps):
	    for fprog in cost_para_programs[idx]:
		prog += fprog(gammas[idx])

	    for fprog in driver_para_programs[idx]:
		prog += fprog(betas[idx])

	return prog

    return psi_ref

def get_analytical_gradient_component_qaoa(cost_ham, ref_ham, psi_ref,
        component_index):
    """
    Maps: Parameterized_Program -> Parameterized_Program

    :returns: a function
    """
    prog = pq.Program()
    psi_ref_lower = psi_ref[:component_index-1] #Need to get the underlying programs
    psi_ref_upper = psi_ref[component_index:] #Need to get the underlying programs
    analytical_gradients_list = []
    ancilla_qubit_index = len(prog.qubit_register) #Check how to add a new qubit
    prog += psi_ref_lower #Fix!
    prog.inst(H(ancilla_qubit_index))
    prog.inst(CNOT(component_index, ancilla_qubit_index)) #Check the order on this!
    prog += psi_ref_upper #Fix!
    i_one = np.array([[1.0, 0.0], [0.0, 1.0j]])
    prog.defgate("I-ONE", i_one)
    prog.inst("I-ONE", ancilla_qubit_index)
    prog.inst(H(ancilla_qubit_index))
    #Measurement?


if __name__ == "__main__":
    square_ring_edges = [(0,1), (1,2), (2,3), (3,0)]
    graph = edges_to_graph(square_ring_edges)
    cost_ham = get_cost_ham(graph)
    ref_ham = get_ref_ham(graph)
    trotterization_steps = 2 #Referred to as "p" in the paper


