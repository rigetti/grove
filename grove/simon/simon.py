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
Module for  Simon's Algorithm.
For more information, see [Simon1995]_, [Loceff2015]_, [Watrous2006]_

.. [Simon1995] Simon, D.R. (1995), `"On the power of quantum computation"`_,
 35th Annual Symposium on Foundations of Computer Science, Proceedings, p. 116-123.

.. [Loceff2015] Loceff, M. (2015), `"A Course in Quantum Computing for the Community College"`_,
 Volume 1, Chapter 18, p 484-541.

.. [Watrous2006] Watrous, J. (2006), `"Simon's Algorithm"`_, University of Calgary CPSC 519/619:
 Quantum Computation, Lecture 6.

.. _`"On the power of quantum computation"`: https://courses.cs.washington.edu/courses/cse599/01wi/
 papers/simon_qc.pdf

.. _`"A Course in Quantum Computing for the Community College"`: http://lapastillaroja.net/
 wp-content/uploads/2016/09/Intro_to_QC_Vol_1_Loceff.pdf

.. _`"Simon's Algorithm"`: https://cs.uwaterloo.ca/~watrous/CPSC519/LectureNotes/06.pdf
"""

from collections import defaultdict
from operator import xor

import numpy as np
import numpy.random as rd
import pyquil.quil as pq
from pyquil.gates import H

import grove.simon.utils as utils


def create_1to1_bitmap(mask):
    """
    A helper to create a bit map function (as a dictionary) for a given mask. E.g. for a mask
    :math:`m = 10` the return is a dictionary:

    >>> create_1to1_bitmap('10')
    ... {
    ...     '00': '10',
    ...     '01': '11',
    ...     '10': '00',
    ...     '11': '01'
    ... }

    :param String mask: binary mask as a string of 0's and 1's
    :return: dictionary containing a mapping of all possible bit strings of the same length as the
        mask's string and their mapped bit-string value
    :rtype: Dict[String, String]
    """
    n_bits = len(mask)
    form_string = "{0:0" + str(n_bits) + "b}"
    bit_map_dct = {}
    for idx in range(2**n_bits):
        bit_string = form_string.format(idx)
        bit_map_dct[bit_string] = utils.bitwise_xor(bit_string, mask)
    return bit_map_dct


def create_valid_2to1_bitmap(mask, random_seed=None):
    """
    A helper to create a 2-to-1 binary function that is invariant with respect to the application of
    a specified XOR bitmask. This property must be satisfied if a 2-to-1 function is to be used in
    Simon's algorithm

    More explicitly, such a 2-to-1 function :math:`f` must satisfy :math:`f(x) = f(x \oplus m)`
    where :math:`m` is a bit mask and :math:`\oplus` denotes the bit wise XOR operation. An example
    of such a function is the truth-table

    ====  ====
     x    f(x)
    ====  ====
    000   101
    001   010
    010   000
    011   110
    100   000
    101   110
    110   101
    111   010
    ====  ====

    Note that, e.g. both `000` and `110` map to the same value `101` and
    :math:`000 \oplus 110 = 110`. The same holds true for other pairs.

    :param String mask: mask input that defines the periodicity of f
    :param Integer random_seed: (optional) integer to set numpy.random.seed parameter.
    :return: dictionary containing the truth table of a valid 2-to-1 boolean function
    :rtype: Dict[String, String]
    """
    if random_seed is not None:
        rd.seed(random_seed)
    bit_map = create_1to1_bitmap(mask)
    n_samples = int(len(bit_map.keys()) / 2)

    # We create a 2-to-1 mapping and hence need to generate a list with exactly half the possible
    # bit-strings. We do this by randomly sampling the set of all possible bit-strings.
    # Moreover we sort the keys as the return order is not guaranteed and we want to remove
    # this randomness
    range_of_2to1_map = list(rd.choice(list(sorted(bit_map.keys())), replace=False, size=n_samples))

    list_of_bitstring_tuples = sorted([(k, v) for k, v in bit_map.items()], key=lambda x: x[0])

    bit_map_dct = {}
    for cnt in range(n_samples):
        bitstring_tup = list_of_bitstring_tuples[cnt]
        val = range_of_2to1_map[cnt]
        bit_map_dct[bitstring_tup[0]] = val
        bit_map_dct[bitstring_tup[1]] = val

    return bit_map_dct


class Simon(object):
    """
    This class contains an implementation of Simon's algorithm using pyQuil. For more references see
    the documentation_

    .. _documentation: http://grove-docs.readthedocs.io/en/latest/simon.html
    """

    def __init__(self):
        self.unitary_function_mapping = None
        self.n_qubits = None
        self.n_ancillas = None
        self._qubits = None
        self.computational_qubits = None
        self.ancillas = None
        self.simon_circuit = None
        self._dict_of_linearly_indep_bit_vectors = {}
        self.mask = None
        self.bit_map = None
        self.classical_register = None

    def _construct_simon_circuit(self):
        """
        Implementation of the quantum portion of Simon's Algorithm.

        Given a list of input qubits, all initially in the :math:`\\vert 0\\rangle` state, create a
        program that applies the Walsh-Hadamard transform the qubits before and after going through
        the oracle.

        :return: A program corresponding to the desired instance of Simon's Algorithm.
        :rtype: Program
        """
        simon_circuit = pq.Program()

        oracle_name = "SIMON_ORACLE"
        simon_circuit.defgate(oracle_name, self.unitary_function_mapping)

        simon_circuit.inst([H(i) for i in self.computational_qubits])
        simon_circuit.inst(tuple([oracle_name] + sorted(self._qubits, reverse=True)))
        simon_circuit.inst([H(i) for i in self.computational_qubits])
        return simon_circuit

    def _init_attr(self, bitstring_map):
        """
        Acts instead of __init__ method to instantiate the necessary Simon Object state.

        :param Dict[String, String] bitstring_map: truth-table of the input bitstring map in
        dictionary format
        :return: None
        :rtype: NoneType
        """
        self.bit_map = bitstring_map
        self.n_qubits = len(list(bitstring_map.keys())[0])
        self.n_ancillas = self.n_qubits
        self._qubits = list(range(self.n_qubits + self.n_ancillas))
        self.computational_qubits = self._qubits[:self.n_qubits]
        self.ancillas = self._qubits[self.n_qubits:]
        self.unitary_function_mapping, _ = self._compute_unitary_oracle_matrix(bitstring_map)
        self.simon_circuit = self._construct_simon_circuit()
        self._dict_of_linearly_indep_bit_vectors = {}
        self.mask = None

    @staticmethod
    def _compute_unitary_oracle_matrix(bitstring_map):
        """
        Computes the unitary matrix that encodes the orcale function for Simon's algorithm

        :param Dict[String, String] bitstring_map: truth-table of the input bitstring map in
        dictionary format
        :return: a dense matrix containing the permutation of the bit strings and a dictionary
        containing the indices of the non-zero elements of the computed permutation matrix as
        key-value-pairs
        :rtype: Tuple[2darray, Dict[String, String]]
        """
        n_bits = len(list(bitstring_map.keys())[0])

        # We instantiate an empty matrix of size 2 * n_bits to encode the mapping from n qubits
        # to n ancillas, which explains the factor 2 overhead.
        # To construct the matrix we go through all possible state transitions and pad the index
        # according to all possible states the ancilla-subsystem could be in
        ufunc = np.zeros(shape=(2 ** (2 * n_bits), 2 ** (2 * n_bits)))
        index_mapping_dct = defaultdict(dict)
        for b in range(2**n_bits):
            # padding according to ancilla state
            pad_str = np.binary_repr(b, n_bits)
            for k, v in bitstring_map.items():
                # add mapping from initial state to the state in the ancilla system.
                # pad_str corresponds to the initial state of the ancilla system.
                index_mapping_dct[pad_str + k] = utils.bitwise_xor(pad_str, v) + k
                # calculate matrix indices that correspond to the transition-matrix-element
                # of the oracle unitary
                i, j = int(pad_str+k, 2), int(utils.bitwise_xor(pad_str, v) + k, 2)
                ufunc[i, j] = 1
        return ufunc, index_mapping_dct

    def find_mask(self, cxn, bitstring_map):
        """
        Runs Simon's mask_array algorithm to find the mask.

        :param QVMConnection cxn: the connection to the Rigetti cloud to run pyQuil programs
        :param Dict[String, String] bitstring_map: a truth table describing the boolean function,
            whose period is  to be found.

        :return: Returns the mask of the bitstring map or raises an Exception if the mask cannot be
            found.
        :rtype: String
        """
        self._init_attr(bitstring_map)

        # create the samples of linearly independent bit-vectors
        self._sample_independent_bit_vectors(cxn)
        # try to invert the mask and check validity
        self._invert_mask_equation()

        if self._check_mask_correct():
            return self.mask
        else:
            raise Exception("No valid mask found")

    def _sample_independent_bit_vectors(self, cxn):
        """This method samples :math:`n-1` linearly independent vectors using the Simon Circuit.
        It attempts to put the sampled bitstring into a dictionary and only terminates once the
        dictionary contains :math:`n-1` samples

        :param cxn: Connection object to a QVM or QPU
        :return: None
        :rtype: NoneType
        """
        while len(self._dict_of_linearly_indep_bit_vectors) < self.n_qubits - 1:
            sampled_bit_string = np.array(cxn.run_and_measure(self.simon_circuit,
                                                              self.computational_qubits)[0],
                                          dtype=int)
            self._add_to_dict_of_indep_bit_vectors(sampled_bit_string.tolist())

    def _invert_mask_equation(self):
        """
        This method tries to infer the bit mask of the input function from the sampled :math:`n-1`
        linearly independent bit vectors.

        It first finds the value of the missing most-significant bit (MSB) in the collection of
        sampled bit vectors, then constructs a matrix in upper-triangular (row-echelon) form and
        finally uses back-substitution over :math:`GF(2)` to find a solution to the equation

            :math:`\\mathbf{\\mathit{W}}\\mathbf{m}=\\mathbf{a}`

        where :math:`a` represents the bit vector of missing provenance, :math:`\mathbf{m}` is the
        mask to be found and :math:`\\mathbf{\\mathit{W}}` is the constructed upper-triangular
        matrix.

        :return: None
        :rtype: NoneType
        """
        missing_msb = self._add_missing_msb_vector()
        upper_triangular_matrix = np.asarray(
            [tup[1] for tup in sorted(zip(self._dict_of_linearly_indep_bit_vectors.keys(),
                                          self._dict_of_linearly_indep_bit_vectors.values()),
                                      key=lambda x: x[0])])

        msb_unit_vec = np.zeros(shape=(self.n_qubits,), dtype=int)
        msb_unit_vec[missing_msb] = 1

        self.mask = utils.binary_back_substitute(upper_triangular_matrix, msb_unit_vec).tolist()

    def _add_to_dict_of_indep_bit_vectors(self, z):
        """
        This method adds a bit-vector z to the dictionary of independent vectors. It checks the
        provenance (most significant bit) of the vector and only adds it to the dictionary if the
        provenance is not yet found in the dictionary. This guarantees that we can write up a
        resulting matrix in upper-triangular form which by virtue of its form is invertible

        :param z: array containing the bit-vector
        :return: None
        :rtype: NoneType
        """
        if all(np.asarray(z) == 0) or all(np.asarray(z) == 1):
            return None
        msb_z = utils.most_significant_bit(z)

        # try to add bitstring z to samples dictionary directly
        if msb_z not in self._dict_of_linearly_indep_bit_vectors.keys():
            self._dict_of_linearly_indep_bit_vectors[msb_z] = z
        # if we have a conflict with the provenance of a sample try to create
        # bit-wise XOR vector (guaranteed to be orthogonal to the conflict) and add
        # that to the samples.
        # Bail if this doesn't work and continue sampling.
        else:
            conflict_z = self._dict_of_linearly_indep_bit_vectors[msb_z]
            not_z = [xor(conflict_z[idx], z[idx]) for idx in range(len(z))]
            if all(np.asarray(not_z) == 0):
                return None
            msb_not_z = utils.most_significant_bit(not_z)
            if msb_not_z not in self._dict_of_linearly_indep_bit_vectors.keys():
                self._dict_of_linearly_indep_bit_vectors[msb_not_z] = not_z

    def _add_missing_msb_vector(self):
        """
        Finds the missing provenance value in the collection of :math:`n-1` linearly independent
        bit vectors and adds a unit vector corresponding to the missing provenance to the collection

        :return: Missing provenance value as int
        :rtype: Int
        """
        missing_msb = None
        for idx in range(self.n_qubits):
            if idx not in self._dict_of_linearly_indep_bit_vectors.keys():
                missing_msb = idx

        if missing_msb is None:
            raise ValueError("Expected a missing provenance, but didn't find one.")

        augment_vec = np.zeros(shape=(self.n_qubits,))
        augment_vec[missing_msb] = 1
        self._dict_of_linearly_indep_bit_vectors[missing_msb] = augment_vec.astype(int).tolist()
        return missing_msb

    def _check_mask_correct(self):
        """
        Checks if a given mask correctly reproduces the function that was provided to the Simon
        algorithm. This can be done in :math:`O(n)` as it is a simple list traversal.

        :return: True if mask reproduces the input function
        :rtype: Bool
        """
        mask_str = ''.join([str(b) for b in self.mask])
        return all([self.bit_map[k] == self.bit_map[utils.bitwise_xor(k, mask_str)]
                    for k in self.bit_map.keys()])
