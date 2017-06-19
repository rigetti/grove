"""Module for Grover's algorithm.  Uses the quadratic n-qubit controlled gate decomposition from
Barenco et al. (1995): arXiv:quant-ph/9503016. This requires defining O(n) new gates, Makes use
of C. Lavor et al.'s (2003) decomposition of the diffusion operator into elemenary gates :
arXiv:quant-ph/0301079.

Future work should be done to minimize gate count. For instance, linear depth n-qubit Toffoli
gates see Saeedi and Pedram: arXiv:1303.3557. """
import pyquil.quil as pq
from pyquil.quilbase import Qubit
from pyquil.gates import H, X, Z, RZ, MEASURE, STANDARD_GATES
from scipy.linalg import sqrtm
import numpy as np
STANDARD_GATE_NAMES = STANDARD_GATES.keys()


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


def grover(oracle, qubits, query_qubit, num_iter=None):
    """
    Implementation of Grover's Algorithm for a given oracle.

    The query qubit will be left in the zero state afterwards.

    :param oracle: An oracle defined as a Program. It should send |x>|q> to |x>|q \oplus f(x)>,
                   where |q> is a a query qubit, and the range of f is {0, 1}.
    :param qubits: List of qubits for Grover's Algorithm. The last is assumed to be query for the
                   oracle.
    :param query_qubit: The qubit |q> that the oracle write its answer to.
    :param num_iter: The number of iterations to repeat the algorithm for. The default is
                     int(pi(sqrt(N))/4.
    :return: A program corresponding to the desired instance of Grover's Algorithm.
    """
    if len(qubits) < 1:
        raise ValueError("Grover's Algorithm requires at least 1 qubits.")

    if num_iter is None:
        num_iter = int(round(np.pi * 2**(len(qubits) / 2.0 - 2.0)))

    diff_op = diffusion_operator(qubits)
    def_gates = oracle.defined_gates + diff_op.defined_gates
    unique_gates = []
    seen_names = set()
    for gate in def_gates:
        if gate.name not in seen_names:
            seen_names.add(gate.name)
            unique_gates.append(gate)

    many_hadamard = pq.Program().inst(map(H, qubits))
    grover_iter = oracle + many_hadamard + diff_op + many_hadamard
    # To prevent redefining gates, this is not the preferred way.
    grover_iter.defined_gates = []

    prog = pq.Program()
    prog.defined_gates = unique_gates

    # Initialize ancilla to be in the minus state
    prog.inst(X(query_qubit))
    prog.inst(H(query_qubit))

    prog += many_hadamard
    for _ in xrange(num_iter):
        prog += grover_iter

    # Move the ancilla back to the zero state
    prog.inst(H(query_qubit))
    prog.inst(X(query_qubit))
    return prog


def basis_selector_oracle(bitstring, qubits, query_qubit):
    """
    Defines an oracle that selects the ith element of the computational basis.

    Sends the state |x>|q> -> |x>|!q> if x==bitstring and |x>|q> otherwise.

    :param bitstring: The desired bitstring, given as a string of ones and zeros. e.g. "101"
    :param qubits: The qubits the oracle is called on, the last of which is the query qubit. The
                   qubits are assumed to be ordered from most significant qubit to least signicant qubit.
    :param query_qubit: The qubit |q> that the oracle write its answer to.
    :return: A program representing this oracle.
    """
    if len(qubits) != len(bitstring):
        raise ValueError("The bitstring should be the same length as the number of qubits.")
    if not isinstance(query_qubit, Qubit):
        raise ValueError("query_qubit should be a single qubit.")
    if not (isinstance(bitstring, str) and all([num in ('0', '1') for num in bitstring])):
        raise ValueError("The bitstring must be a string of ones and zeros.")
    prog = pq.Program()
    for i, qubit in enumerate(qubits):
        if bitstring[i] == '0':
            prog.inst(X(qubit))

    prog += n_qubit_control(qubits, query_qubit, np.array([[0, 1], [1, 0]]), 'NOT')

    for i, qubit in enumerate(qubits):
        if bitstring[i] == '0':
            prog.inst(X(qubit))
    return prog


if __name__ == "__main__":
    from pyquil.api import SyncConnection
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
    cxn = SyncConnection()
    mem = cxn.run_and_measure(grover_program, [q.index() for q in qubits])
    print mem
