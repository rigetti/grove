"""
Testing the term grouping routines
"""
import pytest
from pyquil.paulis import sI, sX, sY, sZ, PauliSum, PauliTerm
from grove.measurements.term_grouping import (check_trivial_commutation,
                                              commuting_sets_by_indices,
                                              commuting_sets_by_zbasis)


def test_check_trivial_commutation_type():
    """
    Simple type checking to see method complains when the wrong type is passed
    """
    with pytest.raises(TypeError):
        check_trivial_commutation(2, sI(0))


def test_check_trivial_commutation_operation():
    """
    Check if logic is sound
    """
    # everything commutes with the identity
    assert check_trivial_commutation([sI(0)], sX(1))
    assert check_trivial_commutation([sX(0) * sZ(1), sY(2)], sI(0))

    # check if returns false for non-commuting sets
    assert not check_trivial_commutation([sX(0), sX(1)], sZ(0) * sZ(1))

    # check trivial commutation is true
    assert check_trivial_commutation([sX(5) * sX(6)], sZ(4))


def test_commuting_terms_indexed():
    """
    Test performance of commuting_sets_by_index
    """
    pauli_term_1 = PauliSum([sX(0) * sZ(1)])
    pauli_term_2 = PauliSum([sX(1) * sZ(2)])
    pauli_term_3 = PauliSum([sX(0) * sY(3)])
    commuting_set_tuples = commuting_sets_by_indices(
        [pauli_term_1, pauli_term_2, pauli_term_3], check_trivial_commutation)
    correct_tuples = [[(0, 0)], [(1, 0), (2, 0)]]

    assert commuting_set_tuples == correct_tuples


def test_commuting_sets_1():
    ham1 = PauliSum([PauliTerm('Z', i) for i in [0, 1]])
    ham2 = PauliSum([PauliTerm('Z', i) for i in [2, 3, 4]])
    actual = commuting_sets_by_indices([ham1, ham2], check_trivial_commutation)

    desired = [
        [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    ]
    assert actual == desired


def test_commuting_sets_2():
    ham1 = PauliSum([PauliTerm('X', i) for i in [0, 1]])
    ham2 = PauliSum([PauliTerm('X', i) for i in [2, 3, 4]])
    actual = commuting_sets_by_indices([ham1, ham2], check_trivial_commutation)

    desired = [
        [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
    ]
    assert actual == desired


def test_commuting_sets_3():
    ham1 = PauliSum([PauliTerm('Z', i) for i in [0, 1]])
    ham2 = PauliSum([PauliTerm('X', i) for i in [0, 1]])
    actual = commuting_sets_by_indices([ham1, ham2], check_trivial_commutation)

    desired = [
        [(0, 0), (0, 1)],
        [(1, 0), (1, 1)],
    ]
    assert actual == desired


def test_commuting_sets_4():
    ham1 = PauliSum([PauliTerm('Z', 0), PauliTerm('X', 0)])
    ham2 = PauliSum([PauliTerm('X', 1), PauliTerm('Z', 1)])
    actual = commuting_sets_by_indices([ham1, ham2], check_trivial_commutation)

    desired = [
        [(0, 0), (1, 0)],
        [(0, 1), (1, 1)],
    ]
    assert actual == desired


def test_term_grouping():
    """
    Test clumping terms into terms that share the same diagonal basis
    """
    x_term = sX(0) * sX(1)
    z1_term = sZ(1)
    z2_term = sZ(0)
    zz_term = sZ(0) * sZ(1)
    h2_hamiltonian = zz_term + z2_term + z1_term + x_term
    clumped_terms = commuting_sets_by_zbasis(h2_hamiltonian)
    true_set = {('X', 'X'): set([x_term.id()]),
                ('Z', 'Z'): set([z1_term.id(), z2_term.id(), zz_term.id()])}
    for key, value in clumped_terms.items():
        assert set(map(lambda x: x.id(), clumped_terms[key])) == true_set[key]

    zzzz_terms = sZ(1) * sZ(2) + sZ(3) * sZ(4) + \
                 sZ(1) * sZ(3) + sZ(1) * sZ(3) * sZ(4)
    xzxz_terms = sX(1) * sZ(2) + sX(3) * sZ(4) + \
                 sX(1) * sZ(2) * sX(3) * sZ(4) + sX(1) * sX(3) * sZ(4)
    xxxx_terms = sX(1) * sX(2) + sX(2) + sX(3) * sX(4) + sX(4) + \
                 sX(1) * sX(3) * sX(4) + sX(1) * sX(4) + sX(1) * sX(2) * sX(3)
    yyyy_terms = sY(1) * sY(2) + sY(3) * sY(4) + sY(1) * sY(2) * sY(3) * sY(4)

    pauli_sum = zzzz_terms + xzxz_terms + xxxx_terms + yyyy_terms
    clumped_terms = commuting_sets_by_zbasis(pauli_sum)

    true_set = {('Z', 'Z', 'Z', 'Z'): set(map(lambda x: x.id(), zzzz_terms)),
                ('X', 'Z', 'X', 'Z'): set(map(lambda x: x.id(), xzxz_terms)),
                ('X', 'X', 'X', 'X'): set(map(lambda x: x.id(), xxxx_terms)),
                ('Y', 'Y', 'Y', 'Y'): set(map(lambda x: x.id(), yyyy_terms))}
    for key, value in clumped_terms.items():
        assert set(map(lambda x: x.id(), clumped_terms[key])) == true_set[key]

