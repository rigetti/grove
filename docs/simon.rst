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

where the top \\(n\\) qubits \\(\\vert x \\rangle\\) are the input,
and the bottom \\(n\\) qubits \\(\\vert y \\rangle\\) are called ancilla qubits.

The input qubits are prepared with the ancilla qubits into the state
$$(H^{\\otimes n} \\otimes I^{\\otimes n})\\vert 0\\rangle^{\\otimes n}\\vert 0\\rangle^{\\otimes n} = \\vert +\\rangle^{\\otimes n}\\vert 0\\rangle^{\\otimes n}$$
and sent through a blackbox gate \\(U_f\\). Then, the Hadamard-Walsh transform
\\(H^{\\otimes n}\\) is applied to the \\(n\\) input qubits, resulting in the state given by
$$(H^{\\otimes n} \\otimes I^{\\otimes n})U_f\\vert +\\rangle^{\\otimes n}\\vert 0\\rangle^{\\otimes n}$$

It turns out the resulting \\(n\\) input qubits are in a uniform random state
over the space killed by (modulo \\(2\\), bitwise) dot product with \\(s\\).
This covers the one-to-one case as well, if one considers it to be the degenerate \\(s=0\\) case.

Suppose we then measured the \\(n\\) input qubits, calling the bitstring output \\(y\\).
The above property then requires \\(s\\cdot y = 0\\). The space of \\(y\\) that satisfies this is \\(n-1\\) dimensional.
By running this quantum subroutine several times, \\(n-1\\) nonzero linearly independent bitstrings \\(y_i\\), \\(i = 0, \\ldots, n-2\\), can be found, each orthogonal to \\(s\\).

This gives a system of \\(n-1\\) equations, with \\(n\\) unknowns for finding \\(s\\).
One final nonzero bitstring \\(y^{\\prime}\\) can be classically found that is linearly independent to the other \\(y_i\\), but with the property that \\(s\\cdot y^{\\prime} = 1\\).
The combination of \\(y^{\\prime}\\) and the \\(y_i\\) give a system of \\(n\\) independent equations that can then be solved for \\(s\\).

By using a clever implementation of Gaussian Elimination and Back Substitution
for mod-2 equations, as outlined in Reference [2]_,
\\(s\\) can be found relatively quickly. By then
sending separate input states \\(\\vert 0\\rangle\\)
and \\(\\vert s\\rangle\\) through the blackbox \\(U_f\\),
we can find whether or not \\(f(0) = f(s)\\) (in fact,
any pair \\(\\vert x\\rangle\\) and \\(\\vert x\\oplus s\\rangle\\) will do as well). If so, we conclude \\(f\\) is
two-to-one with mask \\(s\\); otherwise, \\(f\\) is one-to-one.

Overall, this algorithm can be solved in \\(O(n^3)\\), i.e., polynomial, time,
whereas the best classical algorithm requires exponential time.


Source Code Docs
----------------

Here you can find documentation for the different submodules in simon.

grove.simon.simon
~~~~~~~~~~~~~~~~~

.. automodule:: grove.simon.simon
    :members:
    :undoc-members:
    :show-inheritance:

.. rubric:: References

.. [1] http://pages.cs.wisc.edu/~dieter/Courses/2010f-CS880/Scribes/05/lecture05.pdf
.. [2] http://lapastillaroja.net/wp-content/uploads/2016/09/Intro_to_QC_Vol_1_Loceff.pdf
