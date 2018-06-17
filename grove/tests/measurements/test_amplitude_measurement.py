"""
Testing the amplitude measurement routines
"""
import pytest
from pyquil.paulis import sX, sY, sZ, sI

from grove.measurements.amplitude_measurement import (
    _single_projector_generator, projector_generator)


def test_single_projector():
    """
    Testing the projector generator.  Test if it returns the correct
    non-hermitian operator. These tests are for a single qubit
    """
    zero_projector = _single_projector_generator(0, 0, 0)
    true_zero_projector = 0.5 * (sZ(0) + sI(0))
    assert zero_projector == true_zero_projector

    with pytest.raises(TypeError):
        _single_projector_generator('a', 0, 5)
    with pytest.raises(TypeError):
        _single_projector_generator(0, 'a', 5)
    with pytest.raises(TypeError):
        _single_projector_generator(0, 0.0, 5)
    with pytest.raises(TypeError):
        _single_projector_generator(0.0, 0, 5)
    with pytest.raises(ValueError):
        _single_projector_generator(5, 0, 5)
    with pytest.raises(ValueError):
        _single_projector_generator(1, 4, 5)

    one_projector = _single_projector_generator(1, 1, 5)
    true_one_projector = 0.5 * (sI(5) - sZ(5))
    assert true_one_projector == one_projector

    lowering_projector = _single_projector_generator(0, 1, 2)
    true_lowering_projector = 0.5 * (sX(2) + 1j * sY(2))
    assert true_lowering_projector == lowering_projector

    raising_projector = _single_projector_generator(1, 0, 2)
    true_raising_projector = 0.5 * (sX(2) - 1j * sY(2))
    assert true_raising_projector == raising_projector


def test_projector_generator():
    """
    Test if we are getting accurate projectors--multiqubit case
    """
    true_zero_projector = 0.5 * (sZ(0) + sI(0))
    zero_projector = projector_generator([0], [0])
    assert true_zero_projector == zero_projector

    one_projector = projector_generator([1], [1])
    true_one_projector = 0.5 * (sI(0) - sZ(0))
    assert true_one_projector == one_projector

    lowering_projector = projector_generator([0], [1])
    true_lowering_projector = 0.5 * (sX(0) + 1j * sY(0))
    assert true_lowering_projector == lowering_projector

    raising_projector = projector_generator([1], [0])
    true_raising_projector = 0.5 * (sX(0) - 1j * sY(0))
    assert true_raising_projector == raising_projector
