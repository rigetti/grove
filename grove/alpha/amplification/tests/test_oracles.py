import pytest

from pyquil.quil import Program

from grove.alpha.amplification.oracles import basis_selector_oracle


def test_basis_selector_oracle():
    """Currently we can't test that the output is correct, so we only test for coverage."""
    prog = Program()
    qubit0 = prog.alloc()
    qubit1 = prog.alloc()
    _ = basis_selector_oracle([0], "0")
    _ = basis_selector_oracle([0, 1], "01")
    _ = basis_selector_oracle([qubit0, qubit1], "11")


def test_bad_input_basis_selector_oracle():
    prog = Program()
    qubit0 = prog.alloc()
    with pytest.raises(ValueError):
        _ = basis_selector_oracle([], "100")
    with pytest.raises(ValueError):
        _ = basis_selector_oracle("qubit", "100")
    with pytest.raises(ValueError):
        _ = basis_selector_oracle(qubit0, "100")
    with pytest.raises(ValueError):
        _ = basis_selector_oracle([qubit0], "foo")
