import pytest
from grove.pyqaoa.maxcut_qaoa import maxcut_qaoa
from grove.pyqaoa.qaoa import QAOA
from pyquil.quil import Program
from pyquil.paulis import PauliTerm, PauliSum
from pyquil.gates import H, X, PHASE, CNOT, RZ
import numpy as np
from mock import Mock, patch
import pyquil.qvm as qvm_mod


def isclose(a, b, rel_tol=1e-10, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def compare_progs(test, reference):
    """
    compares two programs gate by gate, param by param
    """
    tinstr = test.instructions
    rinstr = reference.instructions
    assert len(tinstr) == len(rinstr)
    for idx in xrange(len(tinstr)):
        # check each field of the instruction object
        assert tinstr[idx].operator_name == rinstr[idx].operator_name
        assert len(tinstr[idx].parameters) == len(rinstr[idx].parameters)
        for pp in xrange(len(tinstr[idx].parameters)):
            cmp_val = isclose(tinstr[idx].parameters[pp], rinstr[idx].parameters[pp])
            assert cmp_val

        assert len(tinstr[idx].arguments) == len(rinstr[idx].arguments)
        for aa in xrange(len(tinstr[idx].arguments)):
            assert tinstr[idx].arguments[aa] == rinstr[idx].arguments[aa]


def test_pass_hamiltonians():
    ref_ham = [PauliSum([PauliTerm("X", 0, -1.0)]), PauliSum([PauliTerm("X", 1,
                                                              -1.0)])]
    cost_ham = [PauliTerm("I", 0, 0.5) + PauliTerm("Z", 0, -0.5) *
                PauliTerm("Z", 1, 1.0)]
    fakeQVM = Mock()
    inst = QAOA(fakeQVM, 2, steps=1,
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
        QAOA(fakeQVM, 2, steps=1,
             cost_ham=PauliTerm("X", 0, 1.0), ref_hamiltonian=ref_ham,
             rand_seed=42)


def test_hamiltonians():
    test_graph = [(0, 1)]
    p = 1
    inst = maxcut_qaoa(test_graph, steps=p)

    cost_ops, ref_func = inst.cost_ham, inst.ref_ham
    for op in cost_ops:
        for term in op.terms:
            assert(np.isclose(np.abs(term.coefficient), 0.5))

    assert len(ref_func) == 2
    assert len(cost_ops) == 1


def test_param_prog_p1_barbell():
    test_graph = [(0, 1)]
    p = 1
    with patch('grove.pyqaoa.maxcut_qaoa.qvm_module', spec=qvm_mod) as fakeqvm_mod:
        inst = maxcut_qaoa(test_graph, steps=p)

        param_prog = inst.get_parameterized_program()
        trial_prog = param_prog([1.2, 3.4])
        result_prog = Program().inst([H(0), H(1), X(0), PHASE(1.7)(0), X(0),
                                     PHASE(1.7)(0), CNOT(0, 1), RZ(3.4)(1),
                                     CNOT(0, 1), H(0), RZ(-2.4)(0), H(0), H(1),
                                     RZ(-2.4)(1), H(1)])
        compare_progs(trial_prog, result_prog)


def test_psiref_bar_p2():
    bar = [(0, 1)]
    p = 2
    with patch('grove.pyqaoa.maxcut_qaoa.qvm_module', spec=qvm_mod) as fakeqvm_mod:
        inst = maxcut_qaoa(bar, steps=p)

    param_prog = inst.get_parameterized_program()

    # returns are the rotations correct?
    prog = param_prog([1.2, 3.4, 2.1, 4.5])
    result_prog = Program().inst([H(0), H(1),
                                  X(0), PHASE(1.05)(0), X(0), PHASE(1.05)(0),
                                  CNOT(0, 1), RZ(2.1)(1), CNOT(0, 1),
                                  H(0), RZ(-2.4)(0), H(0),
                                  H(1), RZ(-2.4)(1), H(1),
                                  X(0), PHASE(2.25)(0), X(0), PHASE(2.25)(0),
                                  CNOT(0, 1), RZ(4.5)(1), CNOT(0, 1),
                                  H(0), RZ(-6.8)(0), H(0),
                                  H(1), RZ(-6.8)(1), H(1),
                                  ])
    compare_progs(prog, result_prog)


if __name__ == "__main__":
    test_psiref_bar_p2()
