import numpy as np
import pytest

from grove.alpha.fermion_transforms.bktransform import BKTransform
from grove.alpha.fermion_transforms.jwtransform import JWTransform

"""
Some tests inspired by:
https://github.com/ProjectQ-Framework/FermiLib/blob/develop/src/fermilib/transforms/_bravyi_kitaev_test.py
"""


def test_hardcoded_transform():
    n_qubits = 16
    bkt = BKTransform(n_qubits)
    x = bkt.kill(9)
    y = bkt.create(9)

    assert str(x) == '0.5j*Z7*Y9*X11*X15 + (0.5+0j)*Z7*Z8*X9*X11*X15'
    assert str(y) == '-0.5j*Z7*Y9*X11*X15 + (0.5+0j)*Z7*Z8*X9*X11*X15'


def test_term_length():
    # create/kill operators are two-term
    n_qubits = 16
    bkt = BKTransform(n_qubits)
    assert len(bkt.create(3)) == 2
    assert len(bkt.kill(3)) == 2

    n_qubits = 7
    bkt = BKTransform(n_qubits)
    assert len(bkt.create(3)) == 2
    assert len(bkt.kill(3)) == 2


def test_throw_errors():
    # throw error when creation outside qubit range
    n_qubits = 16
    bkt = BKTransform(n_qubits)
    with pytest.raises(IndexError):
        bkt.kill(-1)
    with pytest.raises(IndexError):
        bkt.kill(16)
    with pytest.raises(IndexError):
        bkt.kill(17)


def test_locality_invariant():
    # for n_qubits = 2**d, c_j Majorana is always log2(N) + 1 local
    n_qubits = 16
    bkt = BKTransform(n_qubits)
    invariant = np.log2(n_qubits) + 1
    for index in range(n_qubits):
        op = bkt.kill(index)
        op_terms = op.terms
        for term in op_terms:
            coeff = term.coefficient
            #  Identify the c Majorana terms by real
            #  coefficients and check their length.
            if not isinstance(coeff, complex):
                assert len(term) == invariant


@pytest.mark.skip(reason="pyQuil Pauli needs matrix operator / eigenspectrum "
                         "functionality")
def test_eigenspectrum():
    # Jordan-Wigner and Bravyi-Kitaev operators should give same eigenspectrum

    # single number operator
    n_qubits = 16
    bkt = BKTransform(n_qubits)
    op_BK = bkt.create(3) * bkt.kill(3)
    jwt = JWTransform()
    op_JW = jwt.create(3) * jwt.kill(3)
    assert np.sort(np.linalg.eigvals(op_BK.matrix())) == \
           np.sort(np.linalg.eigvals(op_JW.matrix()))

    # sum of number operators
    op_BK = 0
    op_JW = 0
    for i in [1, 3, 5]:
        op_BK += bkt.create(i) * bkt.kill(i)
        op_JW += jwt.create(i) * jwt.kill(i)
    assert np.sort(np.linalg.eigvals(op_BK.matrix())) == \
           np.sort(np.linalg.eigvals(op_JW.matrix()))

    # scaled number operator
    op_BK = 3 * bkt.create(3) * bkt.kill(3)
    op_JW = 3 * jwt.create(3) * jwt.kill(3)
    assert np.sort(np.linalg.eigvals(op_BK.matrix())) == \
           np.sort(np.linalg.eigvals(op_JW.matrix()))
