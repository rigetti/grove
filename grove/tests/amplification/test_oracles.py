import pytest
from pyquil.quil import Program

from grove.amplification.oracles import basis_selector_oracle


def test_basis_selector_oracle():
    """Given that the oracle is a 'black box', we shouldn't test its circuits. We should instead
     test that it gives the right result on a basis set of vectors. Currently we aren't implementing
     this behavior in the Grove tests, so we just write tests for code coverage here."""
    prog = Program()
    qubit0 = prog.alloc()
    qubit1 = prog.alloc()
    _ = basis_selector_oracle([0], "0")
    _ = basis_selector_oracle([0, 1], "01")
    _ = basis_selector_oracle([qubit0, qubit1], "11")


def test_bad_input_basis_selector_oracle():
    with pytest.raises(ValueError):
        _ = basis_selector_oracle([], "100")
