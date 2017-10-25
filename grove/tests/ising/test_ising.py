
import pytest
from grove.ising.ising_qaoa import ising_qaoa
from grove.ising.ising_qaoa import QAOA_ising
from grove.ising.ising_qaoa import ising
from grove.pyqaoa.utils import compare_progs
from pyquil.quil import Program
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.gates import H, CNOT, RZ
import numpy as np
from mock import Mock, patch
import pyquil.api as qvm_mod


def test_pass_hamiltonians():
    ref_ham = [PauliSum([PauliTerm("X", 0, -1.0)]), PauliSum([PauliTerm("X", 1,
                                                                        -1.0)])]
    cost_ham = [PauliTerm("I", 0, 0.5) + PauliTerm("Z", 0, -0.5) *
                PauliTerm("Z", 1, 1.0)]
    fakeQVM = Mock()
    inst = QAOA_ising(fakeQVM, 2, steps=1,
                      cost_ham=cost_ham, ref_hamiltonian=ref_ham)

    c = inst.cost_ham
    r = inst.ref_ham
    assert isinstance(c, list)
    assert isinstance(r, list)
    assert isinstance(c[0], PauliSum)
    assert isinstance(r[0], PauliSum)
    assert len(c) == 1
    assert len(r) == 2

    with pytest.raises(TypeError):
        QAOA_ising(fakeQVM, 2, steps=1,
                   cost_ham=PauliTerm("X", 0, 1.0), ref_hamiltonian=ref_ham,
                   rand_seed=42)


def test_get_angles():
    p = 2
    n_qubits = 2
    fakeQVM = Mock()
    with patch('grove.pyqaoa.qaoa.VQE', spec=VQE) as mockVQEClass:
        inst = mockVQEClass.return_value
        result = Mock()
        result.x = [1.2, 2.1, 3.4, 4.3]
        inst.vqe_run.return_value = result
        MCinst = QAOA_ising(fakeQVM, n_qubits, steps=p,
                            cost_ham=[PauliTerm("X", 0)])
        betas, gammas = MCinst.get_angles()
        assert betas == [1.2, 2.1]
        assert gammas == [3.4, 4.3]


def test_hamiltonians():
    test_j = {(0, 1): 1}
    test_h = [1, 1]
    p = 1
    inst = ising_qaoa(test_h, test_j, steps=p)

    cost_ops, ref_func = inst.cost_ham, inst.ref_ham
    for op in cost_ops:
        print op
        assert(np.isclose(np.abs(op.coefficient), 1.0))

    assert len(ref_func) == 2
    assert len(cost_ops) == 3


def test_param_prog_p1_barbell():
    test_j = {(0, 1): 1}
    test_h = [1, 1]
    p = 1
    with patch('grove.ising.ising_qaoa.api', spec=qvm_mod):
        inst = ising_qaoa(test_h, test_j, steps=p)
        trial_prog = inst.circuit(np.hstack(([1.2], [3.4])))
        result_prog = Program().inst([H(0), H(1), CNOT(0, 1), RZ(6.8)(1), CNOT(0, 1),
                                      RZ(6.8)(0), RZ(6.8)(1), H(0), RZ(-2.4)(0),
                                      H(0), H(1), RZ(-2.4)(1), H(1)])
        compare_progs(trial_prog, result_prog)


def test_ising_solve():
    J = {(0, 1): -2, (2, 3): 3}
    h = [1, 1, -1, 1]
    with patch('grove.ising.ising_qaoa.api', spec=qvm_mod):
        solution, min_energy, circuit = ising(h, J, num_steps=4, verbose=False)
    assert list(solution) == [-1, -1, 1, -1]
    assert min_energy == -9
