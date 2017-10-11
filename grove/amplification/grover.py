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
"""Module for Grover's algorithm.
"""

import numpy as np
import pyquil.quil as pq
from pyquil.gates import H
from pyquil.quilbase import Qubit

from grove.amplification.amplification import amplification_circuit


class Grover(object):
    """This class contains an implementation of Grover's algorithm using pyQuil. For a reference
    """
    def __init__(self):
        self.unitary_function_mapping = None
        self.n_qubits = None
        self.n_ancillas = None
        self.grover_circuit = None
        self.bit_map = None

    @staticmethod
    def _compute_grover_oracle_matrix(bitstring_map):
        """Computes the unitary matrix that encodes the oracle function for Grover's algorithm

        :param dict bitstring_map: dict with string keys corresponding to bitstrings, and integer
         values corresponding to the desired phase on the output state.
        :return: a numpy array corresponding to the unitary matrix for oracle for the given
         bitstring_map
        :rtype: numpy.ndarray
        """
        n_bits = len(list(bitstring_map.keys())[0])
        oracle_matrix = np.zeros(shape=(2 ** n_bits, 2 ** n_bits))
        for b in range(2 ** n_bits):
            pad_str = np.binary_repr(b, n_bits)
            phase_factor = bitstring_map[pad_str]
            oracle_matrix[b, b] = phase_factor
        return oracle_matrix

    def _construct_grover_circuit(self):
        """Contructs an instance of Grover's Algorithm, using initialized values.

        :return: None
        :rtype: NoneType
        """
        oracle = pq.Program()
        oracle_name = "GROVER_ORACLE"
        oracle.defgate(oracle_name, self.unitary_function_mapping)
        qubits = [oracle.alloc() for _ in range(int(self.n_qubits / 2))]
        oracle.inst(tuple([oracle_name] + qubits))
        self.grover_circuit = self.oracle_grover(oracle, qubits)

    def _init_attr(self, bitstring_map):
        """Initializes an instance of Grover's Algorithm given a bitstring_map.

        :param dict bitstring_map: dict with string keys corresponding to bitstrings, and integer
         values corresponding to the desired phase on the output state.
        :return: None
        :rtype: NoneType
        """
        self.bit_map = bitstring_map
        self.unitary_function_mapping = self._compute_grover_oracle_matrix(bitstring_map)
        self.n_qubits = self.unitary_function_mapping.shape[0]
        self._construct_grover_circuit()

    def construct_grover_program(self, bitstring_map):
        """Returns an instance of Grover's Algorithm for the given bitstring_map.

        :param dict bitstring_map: dict with string keys corresponding to bitstrings, and integer
         values corresponding to the desired phase on the output state.
        :return: The corresponding instance of Grover's Algorithm.
        :rtype: Program
        """
        self._init_attr(bitstring_map)
        return self.grover_circuit

    @staticmethod
    def oracle_grover(oracle, qubits, num_iter=None):
        """Implementation of Grover's Algorithm for a given oracle. The query qubit will be left in
         the zero state afterwards.

        :param Program oracle: An oracle defined as a Program. It should send |x> to (-1)^f(x)|x>,
                               where the range of f is {0, 1}.
        :param qubits: List of qubits for Grover's Algorithm.
        :type qubits: list[int or Qubit]
        :param int num_iter: The number of iterations to repeat the algorithm for.
                             The default is the integer closest to :math:`\\frac{\\pi}{4}\sqrt{N}`,
                             where :math:`N` is the size of the domain.
        :return: A program corresponding to the desired instance of Grover's Algorithm.
        :rtype: Program
        """
        if num_iter is None:
            num_iter = int(round(np.pi * 2 ** (len(qubits) / 2.0 - 2.0)))

        uniform_superimposer = pq.Program().inst([H(qubit) for qubit in qubits])
        amp_prog = amplification_circuit(uniform_superimposer, oracle, qubits, num_iter)
        return amp_prog
