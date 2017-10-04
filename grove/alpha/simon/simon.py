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
import pyquil.quil as pq
from pyquil.gates import *
import grove.alpha.simon.utils as u
from six.moves import input

#
# def check_two_to_one(cxn, oracle, ancillas, s):
#     """
#     Check if the oracle is one-to-one or two-to-one. The oracle is known
#     to represent either a one-to-one function, or a two-to-one function
#     with mask :math:`s`.
#
#     :param JobConnection cxn: the connection used to run programs
#     :param Program oracle: the oracle to query;
#                            emulates a classical :math:`f(x)`
#                            function as a blackbox.
#     :param list(int) ancillas: the ancillary qubits, where :math:`f(x)`
#                                 is written to by the oracle
#     :param str s: the proposed mask of the function, found by Simon's algorithm
#     :return: true if and only if the oracle represents a function
#              that is two-to-one with mask :math:`s`
#     :rtype: bool
#     """
#     zero_program = oracle
#     mask_program = pq.Program()
#     for i in range(len(s)):
#         if s[i] == '1':
#             mask_program.inst(X(i))
#     mask_program += oracle
#
#     zero_value = cxn.run_and_measure(zero_program, ancillas)[0]
#     mask_value = cxn.run_and_measure(mask_program, ancillas)[0]
#
#     return zero_value == mask_value


class Simon(object):

    def __init__(self, boolean_function_mapping):
        assert isinstance(boolean_function_mapping, list)
        assert u.is_power2(len(boolean_function_mapping))
        # assert boolean_function_mapping.shape[0] == boolean_function_mapping.shape[1]
        # assert u.is_unitary(boolean_function_mapping)

        self.unitary_function_mapping = self._construct_unitary_matrix(boolean_function_mapping)
        self.n_qubits = int(np.log2(self.unitary_function_mapping.shape[0])) - 1
        self.n_ancillas = self.n_qubits
        self._qubits = list(range(self.n_qubits + self.n_ancillas))
        self.log_qubits = self._qubits[:self.n_qubits]
        self.ancillas = self._qubits[self.n_qubits:]

    def _construct_unitary_matrix(self, mappings):
        """
        Creates a unitary transformation that maps each state
        to the values specified in mappings.

        Some (but not all) of these transformations involve a scratch qubit,
        so one is always provided. That is, if given the mapping of :math:`n`
        qubits, the calculated transformation will be on :math:`n + 1` qubits,
        where the zeroth qubit is the scratch bit and the return value
        of the function is left in the qubits that follow.

        :param list(int) mappings: List of the mappings of :math:`f(x)` on
                                   all length :math:`n` in their decimal
                                   representations.
                                   For example, the following mapping:

                                   - :math:`00 \\rightarrow 00`
                                   - :math:`01 \\rightarrow 10`
                                   - :math:`10 \\rightarrow 10`
                                   - :math:`11 \\rightarrow 00`

                                   Would be represented as :math:`[0, 2, 2, 0]`.
                                   Requires mappings to be either one-to-one,
                                   or two-to-one with unique mask :math:`s`,
                                   as specified in Simon's problem.
        :return: Matrix representing specified unitary transformation.
        :rtype: numpy array
        """
        if len(mappings) < 2:
            raise ValueError("function domain must be at least one bit (size 2)")

        n = len(mappings).bit_length() - 1

        if len(mappings) != 2 ** n:
            raise ValueError("mappings must have a length that is a power of two")

        # Strategy: add an extra qubit by default
        # and force the function to be one-to-one
        reverse_mapping = {x: list() for x in range(2 ** n)}

        unitary_funct = np.zeros(shape=(2 ** (n + 1), 2 ** (n + 1)))

        # Fill in what is known so far
        prospective_mask = None
        for j in range(2 ** n):
            i = mappings[j]
            reverse_mapping[i].append(j)
            num_mappings_to_i = len(reverse_mapping[i])
            if num_mappings_to_i > 2:
                raise ValueError("Function must be one-to-one or two-to-one;"
                                 " at least three domain values map to "
                                 + np.binary_repr(i, n))
            if num_mappings_to_i == 2:
                # to force it to be one-to-one, we promote the output
                # to have scratch bit set to 1
                mask_for_i = reverse_mapping[i][0] \
                             ^ reverse_mapping[i][1]
                if prospective_mask is None:
                    prospective_mask = mask_for_i
                else:
                    if prospective_mask != mask_for_i:
                        raise ValueError("Mask is not consistent")
                i += 2 ** n
            unitary_funct[i, j] = 1

        # if one to one, just ignore the scratch bit as it's already unitary
        unmapped_range_values = list(filter(lambda i: len(reverse_mapping[i]) == 0,
                                            reverse_mapping.keys()))
        if len(unmapped_range_values) == 0:
            return np.kron(np.identity(2), unitary_funct[0:2 ** n, 0:2 ** n])

        # otherwise, if two-to-one, fill the array to make it unitary
        # assuming scratch bit will properly be 0
        lower_index = 2 ** n

        for i in unmapped_range_values:
            unitary_funct[i, lower_index] = 1
            unitary_funct[i + 2 ** n, lower_index + 1] = 1
            lower_index += 2

        u.is_unitary(unitary_funct)
        return unitary_funct

    def _construct_oracle(self, gate_name='FUNCT'):
        """
        Given a unitary :math:`U_f` that acts as a function
        :math:`f:\\{0,1\\}^n\\rightarrow \\{0,1\\}^n`, such that

        .. math::

            U_f\\vert x\\rangle = \\vert f(x)\\rangle

        create an oracle program that performs the following transformation:

        .. math::

            \\vert x \\rangle \\vert y \\rangle
            \\rightarrow \\vert x \\rangle \\vert f(x) \\oplus y\\rangle

        where :math:`\\vert x\\rangle` and :math:`\\vert y\\rangle`
        are :math:`n` qubit states and :math:`\\oplus` is bitwise xor.

        Allocates one scratch bit.

        :param 2darray unitary_funct: Matrix representation :math:`U_f` of the
                                      function :math:`f`,
                                      i.e. the unitary transformation
                                      that must be applied to a state
                                      :math:`\\vert x \\rangle`
                                      to get :math:`\\vert f(x) \\rangle`
        :param list(int) qubits: List of qubits that enter as the input
                            :math:`\\vert x \\rangle`.
        :param list(int) ancillas: List of qubits to serve as the ancillary input
                         :math:`\\vert y \\rangle`.
        :param str gate_name: Optional parameter specifying the name of
                              the gate that will represent unitary_funct
        :return: A program that performs the above unitary transformation.
        :rtype: Program
        """

        p = pq.Program()

        inverse_gate_name = gate_name + '-INV'
        scratch_bit = p.alloc()
        bits_for_funct = [scratch_bit] + self.log_qubits

        p.defgate(gate_name, self.unitary_function_mapping)
        p.defgate(inverse_gate_name, np.linalg.inv(self.unitary_function_mapping))
        p.inst(tuple([gate_name] + bits_for_funct))
        p.inst(list(map(lambda qs: CNOT(qs[0], qs[1]), zip(self.log_qubits, self.ancillas))))
        p.inst(tuple([inverse_gate_name] + bits_for_funct))

        p.free(scratch_bit)
        return p

    def _hadamard_walsh_append(self, oracle_program):
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
        p = pq.Program()

        # Apply Hadamard, Unitary function, and Hadamard again
        p.inst(list(map(H, self.log_qubits)))
        p += oracle_program
        p.inst(list(map(H, self.log_qubits)))
        return p

    def find_mask(self, cxn):
        """
        Runs Simon'mask_array algorithm to find the mask.

        :param JobConnection cxn: the connection used to run programs
        :param Program oracle: the oracle to query;
                       emulates a classical :math:`f(x)` function as a blackbox.
        :param list(int) qubits: the input qubits
        :return: a tuple t, where t[0] is the bitstring of the mask,
                 t[1] is the number of iterations that the quantum program was run,
                 and t[2] is said quantum program.
        :rtype: tuple
        """
        simon_program = self._hadamard_walsh_append(self._construct_oracle())

        # Generate n-1 linearly independent vectors
        # that will be orthonormal to the mask mask_array
        # Done so by running the quantum program repeatedly
        # and building up a row-echelon matrix echelon_matrix
        iterations = 0
        echelon_matrix = np.array([], dtype=int)
        while True:
            if len(echelon_matrix) == self.n_qubits - 1:
                break
            z = np.array(cxn.run_and_measure(simon_program, self.log_qubits)[0], dtype=int)
            # attempt to insert z in such a way that
            # echelon_matrix remains row-echelon
            # and all rows are orthogonal to mask_array
            echelon_matrix = self.insert_into_row_echelon_binary_matrix(echelon_matrix, z)
            iterations += 1

        # make the matrix square by inserting a row
        # that maintain echelon_matrix in row-echelon form
        echelon_matrix, insert_row_num = self.make_square_row_echelon(echelon_matrix)

        mask_array = np.zeros(shape=(self.n_qubits,), dtype=int)
        # inserted row is chosen not be orthogonal to mask_array
        mask_array[insert_row_num] = 1

        mask_array = self.binary_back_substitute(echelon_matrix, mask_array)

        mask_string = ''.join(str(x) for x in mask_array)

        return mask_string, iterations, simon_program

    def binary_back_substitute(self, W, s):
        """
        Perform back substitution on a binary system of equations.

        Finds the :math:`\\mathbf{x}` such that
        :math:`\\mathbf{\\mathit{W}}\\mathbf{x}=\\mathbf{s}`,
        where all arithmetic is taken bitwise and modulo 2.

        :param 2darray W: A square :math:`n\\times n` matrix of 0s and 1s,
                  in row-echelon form
        :param 1darray s: An :math:`n\\times 1` vector of 0s and 1s
        :return: The :math:`n\\times 1` vector of 0s and 1s that solves the above
                 system of equations.
        :rtype: 1darray
        """
        # iterate backwards, starting from second to last row for back-substitution
        s_copy = np.array(s)
        n = len(s)
        for row_num in range(n - 2, -1, -1):
            row = W[row_num]
            for col_num in range(row_num + 1, n):
                if row[col_num] == 1:
                    s_copy[row_num] = (s_copy[row_num] + s_copy[col_num]) % 2

        return s_copy

    def make_square_row_echelon(self, W):
        """
        Make :math:`\\mathbf{\\mathit{W}}`
        into a square matrix for Simon's algorithm, satisfying a few criteria.

        :param 2darray W: an :math:`(n-1)\\times n` array of 0s and 1s in
                          row-echelon form such that all rows are orthogonal
                          to some length :math:`n` vector
                          of 0s and 1s :math:`\\mathbf{s}`.
        :return: a two-element tuple. The first element is
                 an :math:`n\\times n` square array identical
                 to :math:`\\mathbf{\\mathit{W}}` except with one row added.
                 That row is chosen to keep :math:`\\mathbf{\\mathit{W}}`
                 in row-echelon form. The second element is the row that
                 the new row is in, where the top row is at index 0.
        :rtype: tuple
        """
        n = len(W) + 1

        # Generate one final vector that is not orthonormal to the mask s
        # can do by adding a vector with a single 1
        # that can be inserted so that diag(W) is all ones
        insert_row_num = 0
        while insert_row_num < n - 1 and W[insert_row_num][insert_row_num] == 1:
            insert_row_num += 1

        new_row = np.zeros(shape=(n,), dtype=int)
        new_row[insert_row_num] = 1
        W = np.insert(W, insert_row_num, new_row, 0)

        return W, insert_row_num

    def insert_into_row_echelon_binary_matrix(self, W, z):
        """
        Given a matrix :math:`\\mathbf{\\mathit{W}}` of 0s and 1s
        in row-echelon form, such that each row is orthogonal to some (unknown)
        vector :math:`\\mathbf{s}` of 0s and 1s, attempt to insert a new row
        into :math:`\\mathbf{\\mathit{W}}` that maintains the above property.

        Besides the above property, a vector :math:`\\mathbf{z}` is given
        that is known to also be orthogonal to :math:`\\mathbf{s}`.

        If (and only if) no such row can be inserted with certainty,
        :math:`\\mathbf{\\mathit{W}}` is return unchanged.

        :param 2darray W: a matrix of 0s and 1s in row-echelon form, with rows all
                          orthogonal to some vector of 0s and 1s.

        :param 1darray z: a vector of 0s and 1s known
                          to be orthogonal to :math:`\\mathbf{s}`
        :return: either the same matrix :math:`\\mathbf{\\mathit{W}}`, unchanged,
                 or :math:`\\mathbf{\\mathit{W}}` with one additional
                 row added that maintains the property described above.
        :rtype: 2darray
        """
        n = len(z)
        while np.any(z != 0):  # while z is not all zeros
            if len(W) == 0:
                W = z
                W = W.reshape(1, n)
                break
            msb_z = u.most_significant_bit(z)

            # Search for a row to insert z into,
            # so that it has an earlier significant bit than the row below
            # and a later one than the row above (when reading left-to-right)
            got_to_end = True
            for row_num in range(len(W)):
                row = W[row_num]
                msb_row = u.most_significant_bit(row)
                # if the row as the same msb as z,
                # set z to the bitwise xor of z and the current row
                # as it will be guaranteed to still be orthogonal to s
                if msb_row == msb_z:
                    z = np.array([z[i] ^ row[i] for i in range(n)])
                    got_to_end = False
                    break
                # if the row has a greater msb than z,
                # then this is the row to z insert above
                elif msb_row > msb_z:
                    W = np.insert(W, row_num, z, 0)
                    got_to_end = False
                    break
            # if z has a greater msb than all rows,
            # insert it to the bottom of the array
            if got_to_end:
                W = np.insert(W, len(W), z, 0)

        return W

    def check_two_to_one(self, cxn, oracle, ancillas, s):
        """
        Check if the oracle is one-to-one or two-to-one. The oracle is known
        to represent either a one-to-one function, or a two-to-one function
        with mask :math:`s`.

        :param JobConnection cxn: the connection used to run programs
        :param Program oracle: the oracle to query;
                               emulates a classical :math:`f(x)`
                               function as a blackbox.
        :param list(int) ancillas: the ancillary qubits, where :math:`f(x)`
                                    is written to by the oracle
        :param str s: the proposed mask of the function, found by Simon's algorithm
        :return: true if and only if the oracle represents a function
                 that is two-to-one with mask :math:`s`
        :rtype: bool
        """
        zero_program = oracle
        mask_program = pq.Program()
        for i in range(len(s)):
            if s[i] == '1':
                mask_program.inst(X(i))
        mask_program += oracle

        zero_value = cxn.run_and_measure(zero_program, ancillas)[0]
        mask_value = cxn.run_and_measure(mask_program, ancillas)[0]

        return zero_value == mask_value


if __name__ == "__main__":
    import pyquil.api as api

    # Read function mappings from user
    n = int(input("How many bits? "))
    assert n > 0, "The number of bits must be positive."
    print("Enter f(x) for the following n-bit inputs:")
    mappings = []
    for i in range(2 ** n):
        val = input(np.binary_repr(i, n) + ': ')
        assert all(list(map(lambda x: x in {'0', '1'}, val))), \
            "f(x) must return only 0 and 1"
        mappings.append(int(val, 2))

    qvm = api.SyncConnection()

    qubits = range(n)
    ancillas = range(n, 2 * n)

    unitary_funct = unitary_function(mappings)
    oracle = oracle_function(unitary_funct, qubits, ancillas)

    s, iterations, simon_program = find_mask(qvm, oracle, qubits)
    two_to_one = check_two_to_one(qvm, oracle, ancillas, s)

    if two_to_one:
        print("The function is two-to-one with mask s = ", s)
    else:
        print("The function is one-to-one")
    print("Iterations of the algorithm: ", iterations)

    if input("Show Program? (y/n): ") == 'y':
        print("----------Quantum Program Used----------")
        print(simon_program)
