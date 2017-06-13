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

'''
Module for amplitude amplification, for use in algorithms such as Grover's
algorithm. 

For more information, see arXiv:quant-ph/0005055.
'''

import pyquil.quil as pq
from pyquil.quilbase import Qubit
from pyquil.gates import H, X, Z, RZ, MEASURE, STANDARD_GATES
from scipy.linalg import sqrtm
import numpy as np
STANDARD_GATE_NAMES = STANDARD_GATES.keys()


# Start with A|0>

def amplify(A, U_w, qubits, init=False):
    
    p = pq.Program()
    
    # Apply inital A if init is true
    if init:
        p = A
    
    # A (2|0><0| - I) A^-1 (I - 2|w><w|) A|0>
    p = p + U_w + A.adjoint() + diffusion_operator(qubits) + A;
    
    return p;
    
def n_qubit_control(controls, target, u, gate_name):
    """
    Returns a controlled u gate with n-1 controls.

    Uses a number of gates quadratic in the number of qubits, and defines a linear number of new
    gates. (Roots and adjoints of u.)

    :param controls: The indices of the qubits to condition the gate on.
    :param target: The index of the target of the gate.
    :param u: The unitary gate to be controlled, given as a numpy array.
    :param gate_name: The name of the gate u.
    :return: The controlled gate.
    """
    def controlled_program_builder(controls, target, target_gate_name, target_gate,
                                   defined_gates=set(STANDARD_GATE_NAMES)):
        zero_projection = np.array([[1, 0], [0, 0]])
        one_projection = np.array([[0, 0], [0, 1]])

        control_true = np.kron(one_projection, target_gate)
        control_false = np.kron(zero_projection, np.eye(2, 2))
        control_root_true = np.kron(one_projection, sqrtm(target_gate))

        controlled_gate = control_true + control_false
        controlled_root_gate = control_root_true + control_false
        assert np.isclose(controlled_gate, np.dot(controlled_root_gate, controlled_root_gate)).all()

        sqrt_name = "SQRT" + target_gate_name
        adj_sqrt_name = "ADJ" + sqrt_name

        # Initialize program and populate with gate information
        p = pq.Program()
        for gate_name, gate in ((target_gate_name, controlled_gate),
                                (sqrt_name, controlled_root_gate),
                                (adj_sqrt_name, np.conj(controlled_root_gate.T))):
            if "C" + gate_name not in defined_gates:
                p.defgate("C" + gate_name, gate)
                defined_gates.add("C" + gate_name)

        if len(controls) == 1:
            p.inst(("C" + target_gate_name, controls[0], target))

        else:
            p.inst(("C" + sqrt_name, controls[-1], target))
            many_toff, new_defined_gates = controlled_program_builder(
                controls[:-1], controls[-1], 'NOT', np.array([[0, 1], [1, 0]]), set(defined_gates))
            p += many_toff
            defined_gates.union(new_defined_gates)

            p.inst(("C" + adj_sqrt_name, controls[-1], target))

            # Don't redefine all of the gates.
            many_toff.defined_gates = []
            p += many_toff
            many_root_toff, new_defined_gates = controlled_program_builder(
                controls[:-1], target, sqrt_name, sqrtm(target_gate), set(defined_gates))
            p += many_root_toff
            defined_gates.union(new_defined_gates)

        return p, defined_gates

    p = controlled_program_builder(controls, target, gate_name, u)[0]
    return p


def diffusion_operator(qubits):
    """Constructs the (Grover) diffusion operator on qubits, assuming they are ordered from most
    significant qubit to least significant qubit.

    The diffusion operator is the diagonal operator given by(1, -1, -1, ..., -1).

    :param qubits: A list of ints corresponding to the qubits to operate on. The operator
                   operates on bistrings of the form |qubits[0], ..., qubits[-1]>.
    """
    p = pq.Program()

    if len(qubits) == 1:
        p.inst(H(qubits[0]))
        p.inst(Z(qubits[0]))
        p.inst(H(qubits[0]))

    else:
        p.inst(map(X, qubits))
        p.inst(H(qubits[-1]))
        p.inst(RZ(-np.pi)(qubits[0]))
        p += n_qubit_control(qubits[:-1], qubits[-1], np.array([[0, 1], [1, 0]]), "NOT")
        p.inst(RZ(-np.pi)(qubits[0]))
        p.inst(H(qubits[-1]))
        p.inst(map(X, qubits))
    return p


if __name__ == "__main__":
    from pyquil.forest import Connection
    import sys
    try:
        target = sys.argv[1]
    except IndexError:
        raise ValueError("Enter a target bitstring for Grover's Algorithm.")

    grover_program = pq.Program()
    qubits = [grover_program.alloc() for _ in target]
    query_qubit = grover_program.alloc()
    oracle = basis_selector_oracle(target, qubits, query_qubit)
    grover_program += grover(oracle, qubits, query_qubit)

    # To instantiate the qubits in the program. This is just a temporary hack.
    grover_program.out()
    print grover_program
    cxn = Connection()
    mem = cxn.run_and_measure(grover_program, [q.index() for q in qubits])
    print mem
