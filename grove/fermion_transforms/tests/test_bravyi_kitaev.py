from grove.fermion_transforms.bktransform import BKTransform
from pybond.bk_hamiltonian import BKHamiltonian

def test_bkh():
    # test rel. BK_Hamiltonian
    n_qubits = 16
    bkt = BKTransform(n_qubits)
    x = bkt.kill(9)
    y = bkt.create(9)

    bkh = BKHamiltonian()
    bkh.n_qubits = 16
    xp = bkh.fermi_to_pauli(1.0, [9], [1])  # kill
    yp = bkh.fermi_to_pauli(1.0, [9], [-1])  # create

    assert str(x) == str(xp)
    assert str(y) == str(yp)
    assert str(x) == '0.5j*Z7*Y9*X11*X15 + 0.5*Z7*Z8*X9*X11*X15'
    assert str(y) == '-0.5j*Z7*Y9*X11*X15 + 0.5*Z7*Z8*X9*X11*X15'

    # test rel. BK_Hamiltonian


    # test rel. power-of-2 Bravyi-Kitaev
