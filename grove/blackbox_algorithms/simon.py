"""Module for Simon's Algorithm."""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np
import utils as bbu
from blackbox import AbstractBlackBoxAlgorithm

class Simon(AbstractBlackBoxAlgorithm):
    def __init__(self, n, mappings):
        """
        Creates an instance of an object to simulate the Bernstein-Vazirani Algorithm
        such that the function f(x) is two-to-one with nonzero mask s,
        i.e. f(x)=f(y) if and only if x=y or x+y=s, where addition is taken to be bitwise XOR.
        :param n: the number of bits in the domain and range of the function
        :param mappings: List of the mappings of f(x) on all length n bitstrings.
                          For example, the following mapping:
                          00 -> 00
                          01 -> 10
                          10 -> 10
                          11 -> 00
                          Would be represented as ['00', '10', '10', '00'].
        """
        func = lambda x: bbu.bitstring_to_integer(mappings[x])
        AbstractBlackBoxAlgorithm.__init__(self, n, n, func)

    def generate_prog(self, oracle):
        """
        Implementation of Simon's Algorithm.
        For given f: {0,1}^n -> {0,1}^n, determine the nonzero mask s of the two-to-one function.
        :param oracle: Program representing unitary application of function.
        """
        p = pq.Program()

        # Apply Hadamard, Unitary function, and Hadamard again
        p.inst(map(H, self._input_bits))
        p += oracle
        p.inst(map(H, self._input_bits))
        return p


if __name__ == "__main__":
    import pyquil.forest as forest

    # Read function mappings from user
    n = int(input("How many bits? "))
    assert n > 0, "The number of bits must be positive."
    print "Enter f(x) for the following n-bit inputs:"
    mappings = []
    for i in range(2 ** n):
        val = raw_input(bbu.integer_to_bitstring(i, n) + ': ')
        assert all(map(lambda x: x in {'0', '1'}, val)), "f(x) must return only 0 and 1"
        mappings.append(val)

    simon = Simon(n, mappings)
    simon_program = simon.get_program()
    qubits = simon.get_input_bits()

    simon_program.out()
    print simon_program
    qvm = forest.Connection()


    # Generate n-1 linearly independent vectors that will be orthonormal to the mask s
    # Done so by running the Simon program repeatedly and building up a row-echelon matrix W
    # See http://lapastillaroja.net/wp-content/uploads/2016/09/Intro_to_QC_Vol_1_Loceff.pdf
    iterations = 0
    W = None
    while True:
        if W is not None and len(W) == n-1:
            break
        z = np.array(qvm.run_and_measure(simon_program, [q.index() for q in qubits])[0])
        iterations += 1
        # attempt to insert into W so that W remains in row-echelon form and all rows are linearly independent
        while np.any(z):  # while it's not all zeros
            if W is None:
                W = z.reshape(1, n)
                break
            msb_z = bbu.most_significant_bit(z)

            # Search for a row to insert z into, so that it has an early significant bit than the row below
            # and a later one than the row above (when reading left-to-right)
            got_to_end = True
            for row_num in range(len((W))):
                row = W[row_num]
                msb_row = bbu.most_significant_bit(row)
                if msb_row == msb_z:
                    # special case: if equality, xor with row for another potential orthonormal vector
                    z = np.array([z[i] ^ row[i] for i in range(n)])
                    got_to_end = False
                    break
                elif msb_row > msb_z:
                    W = np.insert(W, row_num, z, 0)
                    got_to_end = False
                    break
            if got_to_end:
                W = np.insert(W, len(W), z, 0)

    # Generate one final vector that is not orthonormal to the mask s
    # can do by adding a vector with a single 1
    # that can be inserted so that diag(W) is all ones
    insert_row_num = 0
    while insert_row_num < n - 1 and W[insert_row_num][insert_row_num] == 1:
        insert_row_num += 1

    new_row = np.zeros(shape=(n,))
    new_row[insert_row_num] = 1
    W = np.insert(W, insert_row_num, new_row, 0)

    s = np.zeros(shape=(n,))
    s[insert_row_num] = 1

    # iterate backwards, starting from second to last row for back-substitution
    for row_num in range(n-2, -1, -1):
        row = W[row_num]
        for col_num in range(row_num+1, n):
            if row[col_num] == 1:
                s[row_num] = int(s[row_num]) ^ int(s[col_num])

    print "The mask s is ", ''.join([str(int(bit)) for bit in s])
    print "Iterations of the algorithm: ", iterations
    