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
Module for the Bernstein-Vazirani Algorithm.
For more information, see [Loceff2015]_

.. [Loceff2015] Loceff, M. (2015), `"A Course in Quantum Computing for the Community College"`_,
 Volume 1, Chapter 18, p 484-541.

.. _`"A Course in Quantum Computing for the Community College"`: http://lapastillaroja.net/
 wp-content/uploads/2016/09/Intro_to_QC_Vol_1_Loceff.pdf
"""

from collections import defaultdict

import numpy as np
import pyquil.quil as pq
from pyquil.gates import H, X

from grove.bernstein_vazirani import utils


def create_bv_bitmap(dot_product_vector, dot_product_bias):
    """
    This function creates a map from bitstring to function value for a boolean formula :math:`f`
    with a dot product vector :math:`a` and a dot product bias :math:`b`

        .. math::

           f:\\{0,1\\}^n\\rightarrow \\{0,1\\}

           \\mathbf{x}\\rightarrow \\mathbf{a}\\cdot\\mathbf{x}+b\\pmod{2}

           (\\mathbf{a}\\in\\{0,1\\}^n, b\\in\\{0,1\\})

    :param String dot_product_vector: a string of 0's and 1's that represents the dot-product
        partner in :math:`f`
    :param String dot_product_bias: 0 or 1 as a string representing the bias term in :math:`f`
    :return: A dictionary containing all possible bitstring of length equal to :math:`a` and the
        function value :math:`f`
    :rtype: Dict[String, String]
    """
    n_bits = len(dot_product_vector)
    bit_map = {}
    for bit_val in range(2 ** n_bits):
        bit_map[np.binary_repr(bit_val, width=n_bits)] = str(
            (int(utils.bitwise_dot_product(np.binary_repr(bit_val, width=n_bits),
                                           dot_product_vector))
             + int(dot_product_bias, 2)) % 2
        )

    return bit_map


class BernsteinVazirani(object):
    """
    This class contains an implementation of the Bernstein-Vazirani algorithm using pyQuil. For more
    references see the documentation_

    .. _documentation: http://grove-docs.readthedocs.io/en/latest/bernstein_vazirani.html
    """

    def __init__(self):
        self.n_qubits = None
        self.n_ancillas = 1
        self.computational_qubits = None
        self.ancilla = None
        self.bv_circuit = None
        self.solution = None
        self.input_bitmap = None

    @staticmethod
    def _compute_unitary_oracle_matrix(bitstring_map):
        """
        Computes the unitary matrix that encodes the oracle function  used in the Bernstein-Vazirani
        algorithm. It generates a dense matrix for a function :math:`f`

        .. math::

           f:\\{0,1\\}^n\\rightarrow \\{0,1\\}

           \\mathbf{x}\\rightarrow \\mathbf{a}\\cdot\\mathbf{x}+b\\pmod{2}

           (\\mathbf{a}\\in\\{0,1\\}^n, b\\in\\{0,1\\})

        where :math:`(\\cdot)` is the bitwise dot product, that represents the transition-matrix
        elements of the corresponding qubit and ancilla subsystems.

        :param Dict[String, String] bitstring_map: truth-table of the input bitstring map in
        dictionary format
        :return: a dense matrix containing the permutation of the bit strings and a dictionary
        containing the indices of the non-zero elements of the computed permutation matrix as
        key-value-pairs
        :rtype: Tuple[2darray, Dict[String, String]]
        """
        n_bits = len(list(bitstring_map.keys())[0])
        n_ancillas = 1

        # We instantiate an empty matrix of size n_bits + 1 to encode the mapping from n qubits
        # to one ancillas, which explains the additional +1 overhead.
        # To construct the matrix we go through all possible state transitions and pad the index
        # according to all possible states the ancilla-subsystem could be in
        ufunc = np.zeros(shape=(2 ** (n_bits + 1), 2 ** (n_bits + 1)))
        index_mapping_dct = defaultdict(dict)
        for b in range(2**n_ancillas):
            # padding according to ancilla state
            pad_str = np.binary_repr(b, width=1)
            for k, v in bitstring_map.items():
                # add mapping from initial state to the state in the ancilla system.
                # pad_str corresponds to the initial state of the ancilla system.
                index_mapping_dct[pad_str + k] = utils.bitwise_xor(pad_str, v) + k
                # calculate matrix indices that correspond to the transition-matrix-element
                # of the oracle unitary
                i, j = int(pad_str+k, 2), int(utils.bitwise_xor(pad_str, v) + k, 2)
                ufunc[i, j] = 1
        return ufunc, index_mapping_dct

    def _create_bv_circuit(self, bit_map):
        """
        Implementation of the Bernstein-Vazirani Algorithm.

        Given a list of input qubits and an ancilla bit, all initially in the
        :math:`\\vert 0\\rangle` state, create a program that can find :math:`\\vec{a}` with one
        query to the given oracle.

        :param Dict[String, String] bit_map: truth-table of a function for Bernstein-Vazirani with
        the keys being all possible bit vectors strings and the values being the function values
        :rtype: Program
        """
        unitary, _ = self._compute_unitary_oracle_matrix(bit_map)
        full_bv_circuit = pq.Program()

        full_bv_circuit.defgate("BV-ORACLE", unitary)

        # Put ancilla bit into minus state
        full_bv_circuit.inst(X(self.ancilla), H(self.ancilla))

        full_bv_circuit.inst([H(i) for i in self.computational_qubits])
        full_bv_circuit.inst(
            tuple(["BV-ORACLE"] + sorted(self.computational_qubits + [self.ancilla], reverse=True)))
        full_bv_circuit.inst([H(i) for i in self.computational_qubits])
        return full_bv_circuit

    def run(self, cxn, bitstring_map):
        """
        Runs the Bernstein-Vazirani algorithm.

        Given a connection to a QVM or QPU, find the :math:`\\mathbf{a}` and :math:`b` corresponding
        to the function represented by the oracle function that will be constructed from the
        bitstring map.

        :param Connection cxn: connection to the QPU or QVM
        :param Dict[String, String] bitstring_map: a truth table describing the boolean function,
            whose dot-product vector and bias is to be found
        :rtype: BernsteinVazirani
        """

        # initialize all attributes
        self.input_bitmap = bitstring_map
        self.n_qubits = len(list(bitstring_map.keys())[0])
        self.computational_qubits = list(range(self.n_qubits))
        self.ancilla = self.n_qubits  # is the highest index now.

        # construct BV circuit
        self.bv_circuit = self._create_bv_circuit(bitstring_map)

        # find vector by running the full bv circuit
        results = cxn.run_and_measure(self.bv_circuit, self.computational_qubits)
        bv_vector = results[0][::-1]

        # To get the bias term we skip the Walsh-Hadamard transform
        results = cxn.run_and_measure(self.bv_circuit, [self.ancilla])
        bv_bias = results[0][0]

        self.solution = ''.join([str(b) for b in bv_vector]), str(bv_bias)
        return self

    def get_solution(self):
        """
        Returns the solution of the BV algorithm

        :return: a tuple of string corresponding to the dot-product partner vector and the bias term
        :rtype: Tuple[String, String]
        """
        if self.solution is None:
            raise AssertionError("You need to `run` this algorithm first")
        return self.solution

    def check_solution(self):
        """
        Checks if the the found solution correctly reproduces the input.

        :return: True if solution correctly reproduces input bitstring map
        :rtype: Bool
        """
        if self.solution is None:
            raise AssertionError("You need to `run` this algorithm first")
        assert_map = create_bv_bitmap(*self.solution)
        return all([assert_map[k] == v for k, v in self.input_bitmap.items()])
