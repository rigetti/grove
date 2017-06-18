from grove.fermion_transforms.jwtransform import JWTransform


def test_create():
    """
    Testing creation operator produces 0.5 * (X - iY)
    """
    jw = JWTransform()
    creation = jw.create(5)
    # we really need PauliSum and PauliTerm logical compare
    assert '0.5*Z0*Z1*Z2*Z3*Z4*X5 + -0.5j*Z0*Z1*Z2*Z3*Z4*Y5' == creation.__str__()


def test_annihilation():
    """
    Testing creation operator produces 0.5 * (X - iY)
    """
    jw = JWTransform()
    annihilation = jw.kill(5)
    # we really need PauliSum and PauliTerm logical compare
    assert '0.5*Z0*Z1*Z2*Z3*Z4*X5 + 0.5j*Z0*Z1*Z2*Z3*Z4*Y5' == annihilation.__str__()


def test_hopping():
    """
    Hopping term tests
    """
    jw = JWTransform()
    hopping = jw.create(2)*jw.kill(0) + jw.create(0)*jw.kill(2)
    assert '(0.5+0j)*X0*Z1*X2 + (0.5+0j)*Y0*Z1*Y2' == hopping.__str__()
