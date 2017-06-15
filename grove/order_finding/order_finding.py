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
Module to find the order of a modular exponential function.
Subroutine for Shor's Algorithm.
"""

import pyquil.quil as pq
import pyquil.forest as forest
import numpy as np
from pyquil.gates import *
from grove.phaseestimation.phase_estimation import controlled
from grove.qft.fourier import qft

def order_finding(a, N):
    """
    Creates program that calculates the order of f(x) = a^x (mod N).

    Creates two qubit registers. Puts first register into state |x>, and uses phase estimation
    with the multiplication operator to put second register into state |f(x)>. Applying the
    quantum fourier transform to the first register and then measuring it yields a multiple
    of the number of bits divided by the order, from which the order can be extracted classically.

    :param int a: Base for exponentiation
    :param int N: Modulus for exponentiation
    :param int num_iters: Number of times to run algorithm, increase for higher accuracy
    :return: A program to calculate the order of f(x)
    :rtype: Program
    """
    assert gcd(a, N) == 1, "The values a and N must be coprime"
    U = multiplication_operator(a, N)

    # The number of bits needed to specify N
    L = int(np.log2(len(U)))

    # Can have as many bits as desired. For accuracy > 3/4, use at least 2L + 3
    t = 2*L + 3

    # instantiate registers
    register_1 = range(t)
    register_2 = range(t, t+L)

    p = pq.Program()

    # put register 1 into superposition over all states |x>
    p.inst(map(H, register_1))

    # put register 2 into state |1>
    p.inst(X(register_2[0]))

    # use phase estimation to apply f(x) to register 2
    for i in register_1:
        if i > 0:
            U = np.dot(U, U)
        cU = controlled(U)
        name = "CONTROLLED-U{0}".format(2 ** i)
        # define the gate
        p.defgate(name, cU)
        # apply it
        p.inst((name, i) + tuple(register_2))

    # qft register 1
    p += qft(register_1)

    return p

def calculate_order(a, N, iteration_scalar=2, verbose=False):
    """
    Calculates the order of f(x) = a^x (mod N) using a quantum subroutine.

    :param int a: Base for exponentiation
    :param int N: Modulus for exponentiation
    :param int iteration_scalar: Proportionality scalar for number of iterations
    :return: Order of f(x)
    :rtype: int
    """

    # The number of bits to represent N
    L = int(np.ceil(np.log2(N + 1)))

    # The number of bits in the first register
    # Can be as large as desired, recommended at least 2*L + 3 for accuracy > 3/4
    t = 2 * L + 3

    if verbose:
        print "Instantiating", t, "qubits into register 1"
        print "Instantiating", L, "qubits into register 2"

    p = order_finding(a, N)

    if verbose:
        print "The following Quil code was generated:"
        print p

    qvm = forest.Connection()

    # running O(log N) iterations
    num_iters = iteration_scalar * L

    if verbose:
        print "Running for O(log N) iterations"

    results = qvm.run_and_measure(p, range(t), num_iters)
    results = filter(lambda x: x != 0, map(binary_to_integer, results))
    gcd_val = reduce(gcd, results)
    return 2**t / gcd_val


def multiplication_operator(a, N):
    """
    Creates the unitary operator U such that U|y> = |a * y> for all y < N.
    Applying phase estimation using this operator on the input state
    |1> is equivalent to modular exponentiation mod N.

    :param int a: Number to multiply the input state by
    :param int N: Modulus for this operation
    :return: Unitary operator for multiplication by a (mod N)
    :rtype: matrix
    """
    L = int(2**np.ceil(np.log2(N + 1)))
    U = np.zeros(shape=(L, L))
    for i in range(L):
        col = np.zeros(L)
        if i < N:
            col[a*i % N] = 1
        else:
            col[i] = 1
        U[i] = col
    assert np.allclose(np.eye(L), U.dot(U.T.conj())), "Operator should be unitary"
    return U.T


def calculate_order_classical(a, N):
    """
    Calculates the order of f(x) = a^x (mod N) classically.

    :param int a: Base for exponentiation
    :param int N: Modulus for exponentiation
    :param int iteration_scalar: Proportionality scalar for number of iterations
    :return: Order of f(x)
    :rtype: int
    """
    r = 1
    while True:
        if a**r % N == 1:
            return r
        r += 1



def binary_to_integer(x):
    x = x[::-1]
    retval = 0
    for i in range(len(x)):
        if x[i]:
            retval += 2**i
    return retval



def gcd(a, b):
    if b > a:
        return gcd(b, a)
    if b == 0:
        return a
    return gcd(b, a % b)



if __name__ == "__main__":


    print "Let's calculate the order of f(x) = a^x (mod N)"
    a = int(input("Pick the base a: "))
    N = int(input("Pick the modulus N: "))

    order_quantum = calculate_order(a, N, verbose=True)
    print "The quantum algorithm returned an order of:", order_quantum

    order_classical = calculate_order_classical(a, N)
    print "The classical algorithm returned an order of:", order_classical

    if order_quantum != order_classical:
        print "Try again with more iterations or higher precision (more qubits in register 1)."




