"""Module for the Bernstein-Vazirani Algorithm."""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np
import utils as bbu
from blackbox import AbstractBlackBoxAlgorithm

class Simon(AbstractBlackBoxAlgorithm):
    def __init__(self, n, m, mappings):
        func = lambda x: bbu.bitstring_to_integer(mappings[x])
        AbstractBlackBoxAlgorithm.__init__(self, n, m, func)

    def generate_prog(self, oracle):
        """
        Implementation of the Bernstein-Vazirani Algorithm.
        For given a in {0,1}^n and b in {0,1}, can determine a with one query to an oracle
        that provides f(x) = a*x+b (mod 2) for x in {0,1}^n.
        :param oracle: Program representing unitary application of function.
        :param qubits: List of qubits that enter as state |x>.
        :param ancilla: Qubit to serve as input |y>.
        :return: A program corresponding to the desired instance of the
                 Bernstein-Vazirani Algorithm.
        :rtype: Program
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

    simon_program = pq.Program()
    qubits = [simon_program.alloc() for _ in range(n)]
    ancillas = [simon_program.alloc() for _ in range(n)]
    scratch_bit = simon_program.alloc()

    simon = Simon(n, n, mappings)
    simon_program = simon.get_program()
    qubits = simon.get_input_bits()

    simon_program.out()
    print simon_program
    qvm = forest.Connection()


    # Generate n-1 linearly independent vectors that will be orthonormal to the mask s
    iterations = 0
    W = None
    while True:
        if W is not None and len(W) == n-1:
            break
        z = np.array(qvm.run_and_measure(simon_program, [q.index() for q in qubits])[0])
        iterations += 1
        while np.any(z): # while it's not all zeros
            if W is None:
                W = z
                W = W.reshape(1, n)
                break
            msb_z = bbu.most_siginifcant_bit(z)

            got_to_end = True
            for row_num in range(len((W))):
                row = W[row_num]
                msb_row = bbu.most_siginifcant_bit(row)
                if msb_row == msb_z:
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
    insert_row_num = 0
    while insert_row_num < n - 1 and W[insert_row_num][insert_row_num] == 1:
        insert_row_num += 1

    new_row = np.zeros(shape=(n,))
    new_row[insert_row_num] = 1
    W = np.insert(W, insert_row_num, new_row, 0)

    s = np.zeros(shape=(n,))
    s[insert_row_num] = 1

    # iterate backwards, starting from second to last row
    for row_num in range(n-2, -1, -1):
        row = W[row_num]
        for col_num in range(row_num+1, n):
            if row[col_num] == 1:
                s[row_num] = int(s[row_num]) ^ int(s[col_num])

    print "The mask s is ", ''.join([str(int(bit)) for bit in s])
    print "Iterations of the algorithm: ", iterations
    