==================================
Shor's Algorithm and Order Finding
==================================

Overview
--------

In 1994 Peter Shor created a quantum polynomial time algorithm for the 
prime factorization of an integer :math:`N = pq`, where :math:`p` and :math:`q` are prime factors.
This algorithm was truly revolutionary for the field of quantum computing,
for its ability to efficiently solve a problem that is thought to be very
difficult for classical computers. It achieves this task through the use
of reducing prime factorization to the problem of calculating the period
of a function, which can be done efficiently on a quantum computer [1]_.

Classical Computation
_____________________

Shor's algorithm is a prime example of a hybrid classical/quantum algorithm,
which means that is has both classical and quantum steps [2]_. The algorithm starts
with the following classical steps, modified from the original algorithm to deal
with certain edge cases and integers composed of more than two primes [3]_:

1. Check if :math:`N` is either trivial (such as 1, 2, or 3), or invalid (0 or negative)
2. Create a queue of integers to factor, starting with :math:`[N]`
3. While that queue has primes (checked using elliptical curve primality testing and Miller-Rabin):

    3a. Remove a non-prime value from the queue and call it :math:`N`.
    
    3b. Pick a random integer :math:`a` which is less than :math:`N`.
    
    3c. Calculate the GCD of :math:`a` and :math:`N`. If this value is not 1, then append the result of the GCD and :math:`N/GCD` to the queue, and go back to 3a.
    
    3d. Otherwise, calculate the order of :math:`a^x \mod N`, using the quantum subroutine mentioned in the next session.
    
    3e. Using the answer from 3d, calculate the GCD of :math:`a^{r/2} + 1` and :math:`N`, and take that to be the value :math:`g`.
    
    3f. Check that :math:`g` and :math:`g/N` are both integers; if so, add them to the queue. Otherwise, start back at 3a.
    
4. Once all elements in the queue are prime, remove any repeated 1's and return the queue as a list of prime factors for the original :math:`N`.

Order Finding - Quantum Subroutine
__________________________________

As you can see from the previous section, there is a quantum subroutine needed to find the periodicity of the function :math:`a^x \mod N`. **Note that this subroutine does sometime find the incorrect order, due to certain edge cases caused by specific inputs; you should always check that the returned order does indeed satisfy the parameters for the problem**. Luckily, the following quantum algorithm can do this efficiently:

1. Instantiate two registers, with the first register initialized to the uniform superposition of all possible states, and the second register instantiated to the value 1.
2. Apply an oracle from register 1 onto register 2, where the oracle computes :math:`a^x \mod N`
3. Apply the inverse QFT to the first register, and measure to get a multiple of :math:`s/r`, which can be used to find the final value of :math:`r`

The following circuit diagram shows this subroutine [2]_:

.. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Shor%27s_algorithm.svg/369px-Shor%27s_algorithm.svg.png
   :align: center

Source Code Docs
----------------

Here you can find documentation for the different submodules used in Shor's algorithm.

grove.order_finding.shor
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: grove.order_finding.shor
    :members:
    :undoc-members:
    :show-inheritance:

grove.order_finding.order_finding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: grove.order_finding.order_finding
    :members:
    :undoc-members:
    :show-inheritance:

.. rubric:: References

.. [1] Nielsen, Michael A., and Isaac L. Chuang. Quantum Computation and Quantum Information. Cambridge University Press, 2010.

.. [2] https://en.wikipedia.org/wiki/Shor's_algorithm

.. [3] https://github.com/vontell/qstudio/blob/master/src/main/java/core/algorithms/ClassicalImpl.java


