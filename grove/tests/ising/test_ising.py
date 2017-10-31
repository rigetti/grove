from grove.ising.ising_qaoa import ising_qaoa
from grove.pyqaoa.utils import compare_progs
from pyquil.quil import Program
from pyquil.gates import H, CNOT, RZ
import numpy as np
from mock import patch
import pyquil.api as qvm_mod


def test_hamiltonians():
    test_j = {(0, 1): 1}
    test_h = [1, 1]
    p = 1
    inst = ising_qaoa(test_h, test_j, steps=p)
    cost_ops, ref_func = inst.cost_ham, inst.ref_ham
    for op in cost_ops:
        for term in op.terms:
            assert(np.isclose(np.abs(term.coefficient), 1.0))

    assert len(ref_func) == 2
    assert len(cost_ops) == 3


def test_param_prog():
    test_j = {(0, 1): 1}
    test_h = [1, 1]
    p = 1
    with patch('grove.ising.ising_qaoa.api', spec=qvm_mod):
        inst = ising_qaoa(test_h, test_j, steps=p)
        param_prog = inst.get_parameterized_program()
        trial_prog = param_prog([1.2, 3.4])
        result_prog = Program().inst([H(0), H(1), CNOT(0, 1), RZ(6.8)(1), CNOT(0, 1),
                                      RZ(6.8)(0), RZ(6.8)(1), H(0), RZ(-2.4)(0),
                                      H(0), H(1), RZ(-2.4)(1), H(1)])
        compare_progs(trial_prog, result_prog)
