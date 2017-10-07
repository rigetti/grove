from itertools import product

from grove.alpha.fermion_transforms.jwtransform import JWTransform


def test_create():
    """
    Testing creation operator produces 0.5 * (X - iY)
    """
    jw = JWTransform()
    creation = jw.create(5)
    # we really need PauliSum and PauliTerm logical compare
    assert '(0.5+0j)*Z0*Z1*Z2*Z3*Z4*X5 + -0.5j*Z0*Z1*Z2*Z3*Z4*Y5' == creation.__str__()


def test_annihilation():
    """
    Testing creation operator produces 0.5 * (X - iY)
    """
    jw = JWTransform()
    annihilation = jw.kill(5)
    # we really need PauliSum and PauliTerm logical compare
    assert '(0.5+0j)*Z0*Z1*Z2*Z3*Z4*X5 + 0.5j*Z0*Z1*Z2*Z3*Z4*Y5' == annihilation.__str__()


def test_hopping():
    """
    Hopping term tests
    """
    jw = JWTransform()
    hopping = jw.create(2)*jw.kill(0) + jw.create(0)*jw.kill(2)
    assert '(0.5+0j)*X0*Z1*X2 + (0.5+0j)*Y0*Z1*Y2' == hopping.__str__()


def test_multi_ops():
    """
    test construction of Paulis for product of second quantized operators
    """
    jw = JWTransform()
    # test on one particle density matrix
    for p, q, in product(range(6), repeat=2):
        truth = jw.create(p)*jw.kill(q)
        prod_ops_out = jw.product_ops([p, q], [-1, 1])
        assert truth.__str__() == prod_ops_out.__str__()

    # test on two particle density matrix
    for p, q, r, s in product(range(4), repeat=4):
        truth = jw.create(p)*jw.create(q)*jw.kill(s)*jw.kill(r)
        prod_ops_out = jw.product_ops([p, q, s, r], [-1, -1, 1, 1])
        assert truth.__str__() == prod_ops_out.__str__()
