Bernstein-Vazirani Algorithm
============================

Overview
--------
This module emulates the Bernstein-Vazirani Algorithm.

The problem is summarized as follows. Given a function \\(f\\) such that

$$f:\\{0,1\\}^n \\rightarrow \\{0,1\\} \\\\
\\mathbf{x} \\rightarrow \\mathbf{a}\\cdot\\mathbf{x} + b\\pmod{2} \\\\
(\\mathbf{a}\\in\\{0,1\\}^n, b\\in\\{0,1\\})$$

determine \\(\\mathbf{a}\\) and \\(b\\) with as few queries to \\(f\\) as possible.

Classically, \\(n+1\\) queries are required: \\(n\\) for \\(\\mathbf{a}\\) and one for \\(b\\).
However, using a quantum algorithm, only \\(2\\) queries are required: just one each both \\(\\mathbf{a}\\) and \\(b\\).

This module is able to generate and run a program to determine \\(\\mathbf{a}\\) and \\(b\\), given an oracle.
It also has the ability to prescribe a way to generate an oracle out of quantum circuit components, given \\(\\mathbf{a}\\) and \\(b\\).

More details about the Bernstein-Vazirani Algorithm can be found in reference [1]_.

Source Code Docs
----------------

Here you can find documentation for the different submodules in bernstein_vazirani.

grove.bernstein_vazirani.bernstein_vazirani
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: grove.bernstein_vazirani.bernstein_vazirani
    :members:
    :undoc-members:
    :show-inheritance:

.. rubric:: References

.. [1] http://pages.cs.wisc.edu/~dieter/Courses/2010f-CS880/Scribes/04/lecture04.pdf
