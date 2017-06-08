"""Module for the Bernstein-Vazirani Algorithm."""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np
import utils as bbu
from blackbox import AbstractBlackBoxAlgorithm

class BernsteinVaziraniAlgorithm(AbstractBlackBoxAlgorithm):
    def __init__(self, n, m, vec_a, b):
        func = lambda x: (int(np.dot(vec_a, bbu.bitstring_to_array(bbu.integer_to_bitstring(x, len(vec_a))))) + b) % 2
        AbstractBlackBoxAlgorithm.__init__(self, n, m, func)

    def generate_prog(self, oracle):
        """
        Implementation of the Bernstein-Vazirani Algorithm.
        For given a in {0,1}^n and b in {0,1}, can determine a with one query to an oracle
        that provides f(x) = a*x+b (mod 2) for x in {0,1}^n.
        :param oracle: Program representing unitary application of function.
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
    import sys

    if len(sys.argv) != 3:
        raise ValueError("Use program as: python bernstein_vazirani.py vec_a b")
    bitstring = sys.argv[1]
    if not (all([num in ('0', '1') for num in bitstring])):
        raise ValueError("The bitstring must be a string of ones and zeros.")
    if not sys.argv[2] in {'0', '1'}:
        raise ValueError("b must be 0 or 1.")

    vec_a = bbu.bitstring_to_array(bitstring)
    b = int(sys.argv[2])

    n = len(vec_a)

    bv = BernsteinVaziraniAlgorithm(n, 1, vec_a, b)
    bv_program = bv.get_program()
    qubits = bv.get_input_bits()

    bv_program.out()
    print bv_program
    qvm = forest.Connection()
    results = qvm.run_and_measure(bv_program, [q.index() for q in qubits])
    print "The bitstring a is given by: ", "".join(map(str, results[0]))