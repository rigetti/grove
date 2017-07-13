##############################################################################
# Copyright 2016-2017 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

import sys, os
#Redesign the relative import system
#Add the analytical gradient directory
dirname = os.path.dirname(os.path.abspath(__file__))
analytical_gradient_dir = os.path.join(dirname, '../analytical_gradient')
qaoa_dir = os.path.join(dirname, '..')
sys.path.append(analytical_gradient_dir)
sys.path.append(qaoa_dir)

import numpy as np
import pyquil.quil as pq
import pyquil.api as api
from pyquil.gates import *
from pyquil.paulis import *

import analytical_gradient
import utils
import expectation_value

#What Graphs should I write test cases for?

def test_generate_make_controlled():
    ancilla_qubit_index = 1
    make_controlled = analytical_gradient.generate_make_controlled(1)
    pauli_operator = [0, "X"]
    mapped_gate = make_controlled(pauli_operator)
    comparison_gate = CNOT(1,0)
    assert mapped_gate == comparison_gate

def test_pauli_term_to_program():
    pauli_term = PauliTerm("X", 0, 1.0)*PauliTerm("X", 1, 1.0)
    make_controlled = analytical_gradient.generate_make_controlled(2)
    program = analytical_gradient.pauli_term_to_program(pauli_term,
        make_controlled)
    comparison_program = pq.Program().inst(CNOT(2,0), CNOT(2,1))
    utils.compare_progs(program, comparison_program)

def test_hamiltonian_to_programs():
    hamiltonian = (PauliTerm("X", 0, 1.0)*PauliTerm("X", 1, 1.0) +
                   PauliTerm("Z", 0, 1.0)*PauliTerm("Z", 1, 1.0))
    make_controlled = analytical_gradient.generate_make_controlled(2)
    programs = analytical_gradient.hamiltonian_to_programs(hamiltonian,
        make_controlled)
    comparison_programs = [pq.Program().inst(CNOT(2, 0), CNOT(2, 1)),
        pq.Program().inst(CPHASE(np.pi)(2, 0), CPHASE(np.pi)(2, 1))]
    for idx in xrange(len(programs)):
        utils.compare_progs(programs[idx], comparison_programs[idx])

def test_analytical_gradient_expectation_value():
    graph_edges = [(0,1)]
    steps = 1
    betas = [1.3]
    gammas = [1.2]

    graph = maxcut_qaoa_core.edges_to_graph(graph_edges)
    num_qubits = len(graph.nodes())
    reference_state_program = maxcut_qaoa_core.construct_reference_state_program(num_qubits)
    cost_hamiltonian = maxcut_qaoa_core.get_cost_hamiltonian(graph)
    driver_hamiltonian = maxcut_qaoa_core.get_driver_hamiltonian(graph)
    cost_unitary_list = maxcut_qaoa_core.get_program_parameterizer(
        steps, cost_hamiltonian)(gammas)
    driver_unitary_list = maxcut_qaoa_core.get_program_parameterizer(
        steps, driver_hamiltonian)(betas)

    #maxcut_qaoa_unitary_program = maxcut_qaoa_core.zip_programs_lists(
    #    [cost_unitary_list, driver_unitary_list])
    full_program = reference_state_program + maxcut_qaoa_unitary_program
    qvm_connection = api.SyncConnection()
    numerical_expectation_value = expectation_value.expectation(full_program,
        cost_hamiltonian, qvm_connection)

    analytical_expectation_value = np.sin(2*betas[0])*np.sin(gammas[0])
    assert round(analytical_expectation_value, 6) == round(numerical_expectation_value,6)


if __name__ == "__main__":
    test_generate_make_controlled()
    test_pauli_term_to_program()
    test_hamiltonian_to_programs()
