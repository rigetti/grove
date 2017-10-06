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

"""Module for the Simon's Algorithm.
For more information, see

- http://lapastillaroja.net/wp-content/uploads/2016/09/Intro_to_QC_Vol_1_Loceff.pdf
- http://pages.cs.wisc.edu/~dieter/Courses/2010f-CS880/Scribes/05/lecture05.pdf
"""

import numpy as np
import numpy.random as rd
import pyquil.quil as pq
from pyquil.gates import CNOT, H, X, I
import grove.alpha.simon.utils as u
from collections import Counter, defaultdict


def create_periodic_1to1_bitmap(mask):
    n_bits = len(mask)
    form_string = "{0:0" + str(n_bits) + "b}"
    dct = {}
    for idx in range(2**n_bits):
        bit_string = form_string.format(idx)
        dct[bit_string] = u.bit_masking(bit_string, mask)
    return dct


def create_valid_2to1_bitmap(mask):
    bm = create_periodic_1to1_bitmap(mask)
    n_samples = int(len(list(bm.keys())) / 2)
    list_of_half_size = list(rd.choice(list(bm.keys()), replace=False, size=n_samples))

    list_of_tup = sorted([(k, v) for k, v in bm.items()], key=lambda x: x[0])

    dct ={}
    cnt = 0
    while cnt < n_samples:
        tup = list_of_tup[cnt]
        val = list_of_half_size[cnt]
        dct[tup[0]] = val
        dct[tup[1]] = val
        cnt += 1

    return dct


class Simon(object):

    def __init__(self):
        self.unitary_function_mapping = None
        self.n_qubits = None
        self.n_ancillas = None
        self._qubits = None
        self.log_qubits = None
        self.ancillas = None
        self.simon_circuit = None
        self.oracle_circuit = None
        self._dict_of_linearly_indep_bit_vectors = {}
        self.mask = None
        self.bit_map = None
        self.classical_register = None

    # @staticmethod
    # def _construct_unitary_matrix(mappings):
    #     """
    #     Creates a unitary transformation that maps each state
    #     to the values specified in mappings.
    #
    #     Some (but not all) of these transformations involve a scratch qubit,
    #     so one is always provided. That is, if given the mapping of :math:`n`
    #     qubits, the calculated transformation will be on :math:`n + 1` qubits,
    #     where the zeroth qubit is the scratch bit and the return value
    #     of the function is left in the qubits that follow.
    #
    #     :param list(int) mappings: List of the mappings of :math:`f(x)` on
    #                                all length :math:`n` in their decimal
    #                                representations.
    #                                For example, the following mapping:
    #
    #                                - :math:`00 \\rightarrow 10`
    #                                - :math:`01 \\rightarrow 11`
    #                                - :math:`10 \\rightarrow 00`
    #                                - :math:`11 \\rightarrow 01`
    #
    #                                Would be represented as :math:`[2, 3, 0, 1]`.
    #                                Requires mappings to be a one-to-one unique mask :math:`s`,
    #                                as specified in Simon's problem.
    #     :return: Matrix representing specified unitary transformation.
    #     :rtype: numpy array
    #     """
    #     if len(mappings) < 2:
    #         raise ValueError("function domain must be at least one bit (size 2)")
    #
    #     if not u.is_power2(len(mappings)):
    #         raise ValueError("mappings must have a length that is a power of two")
    #
    #     n_bits = len(mappings).bit_length() - 1
    #
    #     # check validity of mapping
    #     c = Counter(mappings)
    #     most_common_map = c.most_common(1)[0]
    #     if most_common_map[1] >= 2:
    #         raise ValueError("Function must be one-to-one;"
    #                          " at least two domain values map to "
    #                          + np.binary_repr(most_common_map[0], n_bits))
    #
    #     unitary_funct = np.zeros(shape=(2 ** n_bits, 2 ** n_bits))
    #     for idx, val in enumerate(mappings):
    #         # TODO: 2-to-1 mask lifting.
    #         unitary_funct[val, idx] = 1
    #
    #     # remember the ordering in the qubit basis...
    #     return np.flipud(unitary_funct)

    # def _construct_oracle(self, gate_name='FUNCT'):
    #     """
    #     Given a unitary :math:`U_f` that acts as a function
    #     :math:`f:\\{0,1\\}^n\\rightarrow \\{0,1\\}^n`, such that
    #
    #     .. math::
    #
    #         U_f\\vert x\\rangle = \\vert f(x)\\rangle
    #
    #     create an oracle program that performs the following transformation:
    #
    #     .. math::
    #
    #         \\vert x \\rangle \\vert y \\rangle
    #         \\rightarrow \\vert x \\rangle \\vert f(x) \\oplus y\\rangle
    #
    #     where :math:`\\vert x\\rangle` and :math:`\\vert y\\rangle`
    #     are :math:`n` qubit states and :math:`\\oplus` is bitwise xor.
    #
    #     Allocates one scratch bit.
    #
    #     :param 2darray unitary_funct: Matrix representation :math:`U_f` of the
    #                                   function :math:`f`,
    #                                   i.e. the unitary transformation
    #                                   that must be applied to a state
    #                                   :math:`\\vert x \\rangle`
    #                                   to get :math:`\\vert f(x) \\rangle`
    #     :param list(int) qubits: List of qubits that enter as the input
    #                         :math:`\\vert x \\rangle`.
    #     :param list(int) ancillas: List of qubits to serve as the ancillary input
    #                      :math:`\\vert y \\rangle`.
    #     :param str gate_name: Optional parameter specifying the name of
    #                           the gate that will represent unitary_funct
    #     :return: A program that performs the above unitary transformation.
    #     :rtype: Program
    #     """
    #
    #     p = pq.Program()
    #
    #     inverse_gate_name = gate_name + '-INV'
    #
    #     p.defgate(gate_name, self.unitary_function_mapping)
    #     p.defgate(inverse_gate_name, np.linalg.inv(self.unitary_function_mapping))
    #
    #     p.inst(tuple([gate_name] + self.log_qubits))
    #     p.inst([CNOT(qb, an) for qb, an in zip(self.log_qubits, self.ancillas)])
    #     p.inst(tuple([inverse_gate_name] + self.log_qubits))
    #
    #     return p

    def _hadamard_walsh_append(self):
        """
        Implementation of the quantum portion of Simon's Algorithm.

        Given a list of input qubits,
        all initially in the :math:`\\vert 0\\rangle` state,
        create a program that applies the Hadamard-Walsh transform the qubits
        before and after going through the oracle.

        :param Program oracle: Program representing unitary application of function
        :param list(int) qubits: List of qubits that enter as the input
                            :math:`\\vert x \\rangle`.
        :return: A program corresponding to the desired instance of
                 Simon's Algorithm.
        :rtype: Program
        """

        h = np.asarray([[1, 1], [1, -1]]) / np.sqrt(2)
        iden = np.asarray(([[1, 0], [0, 1]]))

        had = 1
        for qbt in self.log_qubits:
            had = np.kron(h, had)
        for a in self.ancillas:
            had = np.kron(iden, had)

        p = pq.Program()

        oracle_name = "FUNCT"
        p.defgate(oracle_name, self.unitary_function_mapping)

        hadarmard_name = "HADAMARD"
        p.defgate(hadarmard_name, had)

        # Apply Hadamard, Unitary function, and Hadamard again
        # p.inst([H(i) for i in self.log_qubits])
        # p.inst([I(i) for i in self.ancillas])

        # inverse_gate_name = oracle_name + '-INV'

        # p.defgate(inverse_gate_name, np.linalg.inv(self.unitary_function_mapping))

        p.inst(tuple([hadarmard_name] + sorted(self._qubits, reverse=True)))
        p.inst(tuple([oracle_name] + sorted(self._qubits, reverse=True)))
        p.inst(tuple([hadarmard_name] + sorted(self._qubits, reverse=True)))
        # p.inst(tuple([oracle_name] + self._qubits))
        # p.inst([CNOT(qb, an) for qb, an in zip(self.log_qubits, self.ancillas)])
        # p.inst(tuple([inverse_gate_name] + sorted(self._qubits, reverse=True)))

        # p += self.oracle_circuit
        # print(qvm.wavefunction(p, sa.log_qubits)[0].pretty_print_probabilities())
        # print(qvm.wavefunction(p, sa.log_qubits)[0].pretty_print_probabilities())
        # for an in self.ancillas:
        #     p.measure(an, an)

        # p.inst([H(i) for i in self.log_qubits])
        # p.inst([I(i) for i in self.ancillas])
        # print(p)
        return p

    def _init_attr(self, bitstring_map):
        """Acts instead of __init__ method to instantiate the necessary Simon Object state."""
        self.bit_map = bitstring_map
        self.n_qubits = len(list(bitstring_map.keys())[0])
        self.n_ancillas = self.n_qubits
        self._qubits = list(range(self.n_qubits + self.n_ancillas))
        self.log_qubits = self._qubits[:self.n_qubits]
        self.ancillas = self._qubits[self.n_qubits:]
        self.classical_register = np.asarray(list(range(self.n_qubits + self.n_ancillas)))
        self.unitary_function_mapping, _ = \
            self._compute_unitary_oracle_matrix(bitstring_map)
        # self.oracle_circuit = self._construct_oracle()
        self.simon_circuit = self._hadamard_walsh_append()
        self._reset_attr()

    def _reset_attr(self):
        self._dict_of_linearly_indep_bit_vectors = {}
        self.mask = None
        self.classical_register = np.asarray(list(range(self.n_qubits + self.n_ancillas)))

    @staticmethod
    def _compute_unitary_oracle_matrix(bitstring_map):
        n_bits = len(list(bitstring_map.keys())[0])
        ufunc = np.zeros(shape=(2 ** (2 * n_bits), 2 ** (2 * n_bits)))
        dct = defaultdict(dict)
        for b in range(2**n_bits):
            pad_str = np.binary_repr(b, n_bits)
            for k, v in bitstring_map.items():
                dct[pad_str][pad_str + k] = u.bit_masking(pad_str, v) + k
                i, j = int(pad_str+k, 2), int(u.bit_masking(pad_str, v) + k, 2)
                ufunc[i, j] = 1
        return ufunc, dct

    def find_mask(self, cxn, bitstring_map):
        """
        Runs Simon'mask_array algorithm to find the mask.

        :param JobConnection cxn: the connection used to run programs
        :param Program oracle: the oracle to query;
                       emulates a classical :math:`f(x)` function as a blackbox.
        :param list(int) qubits: the input qubits
        :return: Tuple[Int, List] representing the number of iterations, and the bit mask
        """
        if not isinstance(bitstring_map, dict):
            raise ValueError("Bitstring map needs to be a map from bitstring to bitstring")
        self._init_attr(bitstring_map)

        # if there is no solution yet, reset the solution variables
        self._reset_attr()
        # create the samples of linearly independent bit-vectors
        self._sample_independent_bit_vectors(cxn)
        # try to invert the mask and check validity
        self._invert_mask_equation()

        if self._check_mask_correct():
            return self.mask
        else:
            raise Exception("No valid mask found")

    def _sample_independent_bit_vectors(self, cxn):
        """This method samples n-1 linearly independent vectors that will be orthonormal to the mask
        encoded in the Simon Circuit.
        This is achieved by repeatedly running the circuit and building up a dictionary of linearly
        independent bit-vectors. The key is the provenance of the vector so we can guarantee that
        the resulting matrix is invertible due to the guarantees of an Upper Triangular Matrix

        :param cxn: Connection object to the Quantum Engine (QVM, QPU)
        """
        while len(self._dict_of_linearly_indep_bit_vectors) < self.n_qubits - 1:
            z = np.array(cxn.run_and_measure(self.simon_circuit, self.log_qubits)[0], dtype=int)
            self._add_to_dict_of_indep_bit_vectors(z.tolist())

    def _invert_mask_equation(self):
        """The sampling guarantees that there are n-1 linearly independent vectors based on their
        most significant bit (provenance). This implicates that there is exactly one missing
        provenance value in the sample of independent bit-vectors.

        To reconstruct the mask we find this missing provenance and add a unit-bit-vector :math:`a`
        with the missing provenance to the set of linearly independent bit-vectors. Then we can
        find the mask :math:`\mathbf{m}` by solving the equation

            :math:`\\mathbf{\\mathit{W}}\\mathbf{m}=\\mathbf{a}`
        """
        missing_prov = self._add_missing_provenance_vector()
        upper_triangular_matrix = np.asarray(
            [tup[1] for tup in sorted(zip(self._dict_of_linearly_indep_bit_vectors.keys(),
                                          self._dict_of_linearly_indep_bit_vectors.values()),
                                      key=lambda x: x[0])])

        provenance_unit = np.zeros(shape=(self.n_qubits,), dtype=int)
        provenance_unit[missing_prov] = 1

        # solve matrix equation
        self.mask = u.binary_back_substitute(upper_triangular_matrix, provenance_unit).tolist()

    def _add_to_dict_of_indep_bit_vectors(self, z):
        """
        This method adds a bit-vector z to the dictionary of independent vectors. We keep track of
        this list by ordering them according to their most-significant bit (provenance). This is
        sufficient by virtue of the Gauss elimination procedure.

        :param z: sampled bit-vector
        :return: None
        """
        if all(np.asarray(z) == 0) or all(np.asarray(z) == 1):
            return
        msb_z = u.most_significant_bit(z)

        # try to add bitstring z to samples dictionary directly
        if msb_z not in self._dict_of_linearly_indep_bit_vectors.keys():
            self._dict_of_linearly_indep_bit_vectors[msb_z] = z
        # if we have a conflict with the provenance of a sample try to create
        # bit-wise XOR vector (guaranteed to be orthogonal to the conflict) and add
        # that to the samples.
        # Bail if this doesn't work and continue sampling.
        else:
            conflict_z = self._dict_of_linearly_indep_bit_vectors[msb_z]
            not_z = [conflict_z[idx] ^ z[idx] for idx in range(len(z))]
            if all(np.asarray(not_z) == 0):
                return
            msb_not_z = u.most_significant_bit(not_z)
            if msb_not_z not in self._dict_of_linearly_indep_bit_vectors.keys():
                self._dict_of_linearly_indep_bit_vectors[msb_not_z] = not_z

    def _add_missing_provenance_vector(self):
        """Adds a unit vector with the missing provenance in the collection of independent
        bit-vectors

        :return: Int representing the missing most significant bit / provenance in the collection of
        independent bit-vectors.
        """
        missing_prov = None
        for idx in range(self.n_qubits):
            if idx not in self._dict_of_linearly_indep_bit_vectors.keys():
                missing_prov = idx

        if missing_prov is None:
            raise ValueError("Expected a missing provenance, but didn't find one.")

        augment_vec = np.zeros(shape=(self.n_qubits,))
        augment_vec[missing_prov] = 1
        self._dict_of_linearly_indep_bit_vectors[missing_prov] = augment_vec.astype(int).tolist()
        return missing_prov

    def _check_mask_correct(self):
        mask_str = ''.join([str(b) for b in self.mask])
        return all([self.bit_map[k] == self.bit_map[u.bit_masking(k, mask_str)]
                    for k in self.bit_map.keys()])
