"""Module for the General Black Box Algorithms."""

import pyquil.quil as pq
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
        self._prog += self.generate_prog(oracle)

    def generate_prog(self, oracle):
        raise NotImplementedError("Should be overridden")

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