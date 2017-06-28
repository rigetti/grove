Simon's Algorithm
=================

Overview
--------

This module emulates Simon's Algorithm.

Simon's problem is summarized as follows.
A function \\(f :\\{0,1\\}^n\\rightarrow\\{0,1\\}^n\\) is promised to be either
one-to-one, or two-to-one with some nonzero \\(n\\)-bit mask \\(s\\).
The latter condition means that for any two different \\(n\\)-bit numbers
\\(x\\) and \\(y\\), \\(f(x)=f(y)\\) if and only if \\(x\\oplus y = s\\). The problem
then is to determine whether \\(f\\) is one-to-one or two-to-one, and, if
the latter, what the mask \\(s\\) is, in as few queries to \\(f\\) as possible.

The problem statement and algorithm can be explored further, at a high level,
in reference [1]_. The implementation of the algorithm in this module,
however, follows [2]_.


Algorithm and Details
---------------------
This algorithm involves a quantum component and a classical component.
The quantum part follows similarly to other blackbox oracle algorithms.
First, assume a blackbox oracle \\(U_f\\) is available with the property
$$U_f\\vert x\\rangle\\vert y\\rangle = \\vert x\\rangle\\vert y\\oplus f(x)\\rangle$$

where the top \\(n\\) qubits are the input, and the bottom \\(n\\) qubits are called ancilla qubits.

Prepare the input into the state

$$\\vert +\\rangle^{\\otimes n}\\vert 0\\rangle^{\\otimes n}$$

and send it through a black box gate \\(U_f\\). Apply the Hadamard-Walsh transform
\\(H^{\\otimes n}\\) to the \\(n\\) input qubits, and measure out those \\(n\\) qubits. Call this output \\(y\\).

If the function \\(f\\) is two-to-one with mask \\(s\\), it turns out that the only possible values of \\(y\\) that can be measured
are those that are orthogonal to \\(s\\), i.e. \\(s\\cdot y = 0\\), where \\((\\cdot)\\) is a bitwise dot product, modulo \\(2\\).

By running this algorithm several times, \\(n-1\\) nonzero linearly independent bitstrings \\(y_i\\), \\(i = 0, \\ldots, n-2\\), can be found, each orthogonal to \\(s\\).

One final nonzero bitstring \\(y^{\\prime}\\) can be found that is linearly independent to the other \\(y_i\\), but with the property that \\(s\\cdot y^{\\prime} = 1\\).
This must be true, because there are \\(n\\) linearly independent bitstrings orthogonal to \\(s\\). We have found \\(n-1\\) of these, and the last one,
the all zero bitstring, is not helpful for solving for \\(s\\).

The combination of \\(y^{\\prime}\\) and the \\(y_i\\) give a system of \\(n\\) independent equations that can then be solved for \\(s\\).

By using a clever implementation of Gaussian Elimination and Back Substitution for mod-2 equations,
as outlined in Reference [2]_, \\(s\\) can be found relatively quickly.

Overall, this algorithm can be solved in \\(O(n^3)\\), i.e. polynomial, time, whereas the best classical algorithm
requires exponential time.


Source Code Docs
----------------

Here you can find documentation for the different submodules in phaseestimation.

grove.simon.simon
~~~~~~~~~~~~~~~~~

.. automodule:: grove.simon.simon
    :members:
    :undoc-members:
    :show-inheritance:

.. rubric:: References

.. [1] http://lapastillaroja.net/wp-content/uploads/2016/09/Intro_to_QC_Vol_1_Loceff.pdf
.. [2] http://pages.cs.wisc.edu/~dieter/Courses/2010f-CS880/Scribes/05/lecture05.pdf
