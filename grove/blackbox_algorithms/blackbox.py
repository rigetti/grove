"""Module for general black box algorithms."""
import pyquil.quil as pq
import utils as bbu

class AbstractBlackBoxAlgorithm(object):
    """
    Abstract class for general black box algorithms
    involving functions that map {0,1}^n -> {0,1}^m
    """
    def __init__(self, n, m, func):
        """
        :param n: the number of bits in the domain of the function
        :param m: the number of bits in the range of the function
        :param func: the function that the blackbox oracle is to emulate. While the input space and output spaces are binary strings,
        the function itself is expected to take base 10 integer inputs from 0 to 2^n - 1, inclusive, and output in the space of base 10 integers
        from 0 to 2^m - 1, inclusive
        :return: an object that emulates an algorithm with a black box oracle that maps inputs to outputs according to f: {0,1}^n -> {0,1}^m
        :rtype: AbstractBlackBoxAlgorithm
        """
        self._n = n
        self._m = m
        self._func = func

        self._prog = pq.Program()
        self._input_bits = bbu.get_n_bits(self._prog, self._n)
        self._ancilla_bits = bbu.get_n_bits(self._prog, self._m)

        self._generate_prog()

    def _generate_prog(self):
        """
        Internal helper function that fully instantiates the quantum program to be run. Modifies this own object
        by appending to self._prog.

        The general progression is: from the function properties, generate a unitary that properly acts as the oracle,
        including a number of scratch bits as needed in the most significant bits positions.

        From that, produce the scratch bits and create the oracle itself.

        Finally add gates surrounding the oracle, as needed by the particular algorithm.
        :return: None
        """
        unitary_funct, num_scratch_bits = bbu.unitary_from_function(self._func, self._n, self._m)
        scratch_bits = [self._prog.alloc() for _ in range(num_scratch_bits)]

        oracle = bbu.oracle_function(unitary_funct, self._input_bits, self._ancilla_bits, scratch_bits)
        self._prog += self.generate_prog(oracle)

    def generate_prog(self, oracle):
        """
        An abstract method that fully creates the quantum program, given the oracle. Must be overwritten for specific blackbox algorithm implementations.
        :param oracle: the oracle that emulates func, to be included in the quantum program for querying
        :return: Program object that represents
        :rtype: Program
        """
        raise NotImplementedError("Should be overridden")

    def get_program(self):
        """
        :return: the Quil program that this object represents
        :rtype: Program
        """
        return self._prog

    def num_domain_bits(self):
        """
        :return: the number of bits in the domain of the oracle function
        :rtype: int
        """
        return self._n

    def num_range_bits(self):
        """
        :return: the number of bits in the range of the oracle function
        :rtype: int
        """
        return self._m

    def get_input_bits(self):
        """
        :return: the input qubits
        :rtype: list of Qubit
        """
        return self._input_bits

    def get_ancillia_bits(self):
        """
        :return: the ancilliary qubits
        :rtype: list of Qubit
        """
        return self._ancilla_bits