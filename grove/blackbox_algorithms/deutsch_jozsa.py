"""Module for Deutsch-JozsaAlgorithm."""

import pyquil.quil as pq
from pyquil.gates import *
import utils as bbu
from blackbox import AbstractBlackBoxAlgorithm


class DeutschJozsa(AbstractBlackBoxAlgorithm):
    def __init__(self, n, m, mappings):
        func = lambda x: mappings[x]
        AbstractBlackBoxAlgorithm.__init__(self, n, m, func)

    def generate_prog(self, oracle):
        """
        Implementation of the Deutsch-Jozsa Algorithm.
        Can determine whether a function f mapping {0,1}^n to {0,1} is constant
        or balanced, provided that it is one of them.
        :param oracle: Program representing unitary application of function.
        :param qubits: List of qubits that enter as state |x>.
        :param ancilla: Qubit to serve as input |y>.
        :return: A program corresponding to the desired instance of the
                 Deutsch-Jozsa Algorithm.
        :rtype: Program
        """
        p = pq.Program()

        # Put ancilla bit into minus state
        p.inst(X(self._ancilla_bits[0]), H(self._ancilla_bits[0]))

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
        val = int(raw_input(bbu.integer_to_bitstring(i, n) + ': '))
        assert val in [0,1], "f(x) must return only 0 and 1"
        mappings.append(val)
        
    dj = DeutschJozsa(n, n, mappings)
    deutsch_program = dj.get_program()
    qubits = dj.get_input_bits()

    deutsch_program.out()
    print deutsch_program
    qvm = forest.Connection()

    results = qvm.run_and_measure(deutsch_program, [q.index() for q in qubits])
    print "Results:", results
    print "f(x) is", "balanced" if 1 in results[0] else "constant"