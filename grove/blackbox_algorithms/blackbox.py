"""Module for the General Black Box Algorithms."""

import pyquil.quil as pq
from pyquil.gates import *
import numpy as np
import utils as bbu

class AbstractBlackBoxAlgorithm(object):
    """
    Abstract class for general black box algorithms
    involving functions that map {0,1}^n -> {0,1}^m
    """
    def __init__(self, n, m, func):
        self._n = n
        self._m = m
        self._func = func
        
        self._prog = pq.Program()
        self._input_bits = bbu.get_n_bits(self._prog, self._n)
        self._ancilla_bits = bbu.get_n_bits(self._prog, self._m)
        
        self._generate_prog()
        
    def _generate_prog(self):
        unitary_funct, num_scratch_bits = bbu.unitary_from_function(self._func, self._n, self._m)
        scratch_bits = [self._prog.alloc() for _ in range(num_scratch_bits)]

        oracle = bbu.oracle_function(unitary_funct, self._input_bits, self._ancilla_bits, scratch_bits)
        self._prog += bernstein_vazirani(oracle, qubits, ancillas)

    def num_domain_bits(self):
        return self._n

    def num_range_bits(self):
        return self._m
    
    def get_input_bits(self):
        return self._input_bits
    
    def get_ancillia_bits(self):
        return self._ancilla_bits

    def get_program(self):
        return self._prog

class BernsteinVaziraniAlgorithm(AbstractBlackBoxAlgorithm):
    def __init__(self, vec_a, b):
        pass

        
def bernstein_vazirani(oracle, qubits, ancilla):
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

    # Put ancilla bit into minus state
    p.inst(X(ancilla[0]), H(ancilla[0]))

    # Apply Hadamard, Unitary function, and Hadamard again
    p.inst(map(H, qubits))
    p += oracle
    p.inst(map(H, qubits))
    return p

def bernstein_vazirani_function(vec_a, b):
    def func(x):
        return (int(np.dot(vec_a, bbu.bitstring_to_array(bbu.integer_to_bitstring(x, len(vec_a))))) + b) % 2

    return func

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

    bv_program = pq.Program()
    qubits = [bv_program.alloc() for _ in range(len(vec_a))]
    ancillas = [bv_program.alloc()]

    unitary_funct, num_scratch_bits = bbu.unitary_from_function(bernstein_vazirani_function(vec_a, b), len(vec_a), 1)

    scratch_bits = [bv_program.alloc() for _ in range(num_scratch_bits)]

    oracle = bbu.oracle_function(unitary_funct, qubits, ancillas, scratch_bits)
    bv_program += bernstein_vazirani(oracle, qubits, ancillas)

    print bv_program

    qvm = forest.Connection()
    results = qvm.run_and_measure(bv_program, [q.index() for q in qubits])
    print "The bitstring a is given by: ", "".join(map(str, results[0]))
