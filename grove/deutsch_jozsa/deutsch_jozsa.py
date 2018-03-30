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
"""
Module for the Deutsch-Jozsa Algorithm.
"""
import numpy as np
import pyquil.quil as pq
from pyquil.gates import X, H, CNOT


SWAP_MATRIX = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [0, 0, 0, 1]])
r"""The matrix that performs \alpha\ket{ij}\to\alpha\ket{ji}"""

ORACLE_GATE_NAME = "DEUTSCH_JOZSA_ORACLE"


class DeutschJosza(object):
    def __init__(self):
        self.bit_map = None
        self.unitary_matrix = None
        self.n_qubits = None
        self.n_ancillas = None
        self._qubits = None
        self.computational_qubits = None
        self.deutsch_jozsa_circuit = None

    def is_constant(self, cxn, bitstring_map):
        """Computes whether bitstring_map represents a constant function, given that it is constant
         or balanced. Constant means all inputs map to the same value, balanced means half of the
         inputs maps to one value, and half to the other.

        :param QVMConnection cxn: The connection object to the Rigetti cloud to run pyQuil programs.
        :param bitstring_map: A dictionary whose keys are bitstrings, and whose values are bits
         represented as strings.
        :type bistring_map: Dict[String, String]
        :return: True if the bitstring_map represented a constant function, false otherwise.
        :rtype: bool
        """
        self._init_attr(bitstring_map)
        returned_bitstring = cxn.run_and_measure(self.deutsch_jozsa_circuit, self.computational_qubits)
        # We are only running a single shot, so we are only interested in the first element.
        bitstring = np.array(returned_bitstring, dtype=int)
        constant = all([bit == 0 for bit in bitstring])
        return constant

    def _init_attr(self, bitstring_map):
        """
        Acts instead of __init__ method to instantiate the necessary Deutsch-Jozsa state.

        :param Dict[String, String] bitstring_map: truth-table of the input bitstring map in
        dictionary format, used to construct the oracle in the Deutsch-Jozsa algorithm.
        :return: None
        :rtype: NoneType
        """
        self.bit_map = bitstring_map
        self.n_qubits = len(list(bitstring_map.keys())[0])
        # We use one extra qubit for making the oracle,
        # and one for storing the answer of the oracle.
        self.n_ancillas = 2
        self._qubits = list(range(self.n_qubits + self.n_ancillas))
        self.computational_qubits = self._qubits[:self.n_qubits]
        self.ancillas = self._qubits[self.n_qubits:]
        self.unitary_matrix = self.unitary_function(bitstring_map)
        self.deutsch_jozsa_circuit = self._construct_deutsch_jozsa_circuit()

    def _construct_deutsch_jozsa_circuit(self):
        """
        Builds the Deutsch-Jozsa circuit. Which can determine whether a function f mapping
        :math:`\{0,1\}^n \to \{0,1\}` is constant or balanced, provided that it is one of them.

        :return: A program corresponding to the desired instance of Simon's Algorithm.
        :rtype: Program
        """
        dj_prog = pq.Program()

        # Put the first ancilla qubit (query qubit) into minus state
        dj_prog.inst(X(self.ancillas[0]), H(self.ancillas[0]))

        # Apply Hadamard, Oracle, and Hadamard again
        dj_prog.inst([H(qubit) for qubit in self.computational_qubits])

        # Build the oracle
        oracle_prog = pq.Program()
        oracle_prog.defgate(ORACLE_GATE_NAME, self.unitary_matrix)

        scratch_bit = self.ancillas[1]
        qubits_for_funct = [scratch_bit] + self.computational_qubits
        oracle_prog.inst(tuple([ORACLE_GATE_NAME] + qubits_for_funct))
        dj_prog += oracle_prog

        # Here the oracle does not leave the computational qubits unchanged, so we use a CNOT to
        # to move the result to the query qubit, and then we uncompute with the dagger.
        dj_prog.inst(CNOT(self._qubits[0], self.ancillas[0]))
        dj_prog += oracle_prog.dagger()
        dj_prog.inst([H(qubit) for qubit in self.computational_qubits])
        return dj_prog

    @staticmethod
    def unitary_function(mappings):
        """
        Creates a unitary transformation that maps each state to the values specified
        in mappings.

        Some (but not all) of these transformations involve a scratch qubit, so room for one is
        always provided. That is, if given the mapping of n qubits, the calculated transformation
        will be on n + 1 qubits, where the 0th is the scratch bit and the return value
        of the function is left in the 1st.

        :param mappings: Dictionary of the mappings of f(x) on all length n bitstrings, e.g.

            >>> {'00': '0', '01': '1', '10': '1', '11': '0'}

        :type mappings: Dict[String, Int]
        :return: ndarray representing specified unitary transformation.
        :rtype: np.ndarray
        """
        num_qubits = int(np.log2(len(mappings)))
        bitsum = sum([int(bit) for bit in mappings.values()])

        # Only zeros were entered
        if bitsum == 0:
            return np.kron(SWAP_MATRIX, np.identity(2 ** (num_qubits - 1)))

        # Half of the entries were 0, half 1
        elif bitsum == 2 ** (num_qubits - 1):
            unitary_funct = np.zeros(shape=(2 ** num_qubits, 2 ** num_qubits))
            index_lists = [list(range(2 ** (num_qubits - 1))),
                           list(range(2 ** (num_qubits - 1), 2 ** num_qubits))]
            for j in range(2 ** num_qubits):
                bitstring = np.binary_repr(j, num_qubits)
                value = int(mappings[bitstring])
                mappings.pop(bitstring)
                i = index_lists[value].pop()
                unitary_funct[i, j] = 1
            return np.kron(np.identity(2), unitary_funct)

        # Only ones were entered
        elif bitsum == 2 ** num_qubits:
            x_gate = np.array([[0, 1], [1, 0]])
            return np.kron(SWAP_MATRIX, np.identity(2 ** (num_qubits - 1))).dot(
                np.kron(x_gate, np.identity(2 ** num_qubits)))
        else:
            raise ValueError("f(x) must be constant or balanced")
