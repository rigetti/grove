from grove.fermion_transforms.bktransform import BKTransform


def test_bravyi_kitaev():
    # test drawn from ProjectQ/FermiLib/BKTransform
    n_qubits = 16
    bkt = BKTransform(n_qubits)
    x = bkt.kill(9)
    y = bkt.create(9)

    # proper answer obtained from pybond.bkhamiltonian
    assert str(x) == '0.5j*Z7*Y9*X11*X15 + 0.5*Z7*Z8*X9*X11*X15'
    assert str(y) == '-0.5j*Z7*Y9*X11*X15 + 0.5*Z7*Z8*X9*X11*X15'

if __name__ == '__main__':
    test_bravyi_kitaev()
