"""
Testing the term grouping routines
"""
import pytest
from pyquil.paulis import sI, sX, sY, sZ, PauliSum, PauliTerm
from grove.measurements.term_grouping import (check_trivial_commutation,
                                              commuting_sets_by_indices,
                                              commuting_sets_by_zbasis,
                                              commuting_sets_trivial,
                                              get_diagonalizing_basis,
                                              diagonal_basis_commutes)


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


def test_check_commutation_trivial_grouping_type():
    with pytest.raises(TypeError):
        commuting_sets_trivial(5)


def test_check_commutation_trivial_grouping():
    """
    Check grouping of trivial terms
    """
    commuting_set_one = sX(0) * sZ(1) + sY(2)
    commuting_set_two = PauliSum([sX(1)])
    hamiltonian = commuting_set_one + commuting_set_two
    commuting_sets = commuting_sets_trivial(hamiltonian)
    for comm_set in commuting_sets:
        if len(comm_set) == 2:
            true_set = set(map(lambda x: x.id(), commuting_set_one.terms))
            assert set(map(lambda x: x.id(), comm_set)) == true_set
        elif len(comm_set) == 1:
            true_set = set(map(lambda x: x.id(), commuting_set_two.terms))
            assert set(map(lambda x: x.id(), comm_set)) == true_set


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


def test_term_grouping_weird_term():
    term1 = PauliTerm.from_list([('X', 1), ('Z', 2), ('Y', 3), ('Y', 5),
                                 ('Z', 6), ('X', 7)],
                                coefficient=0.012870253243021476)

    term2 = PauliTerm.from_list([('Z', 0), ('Z', 6)],
                                coefficient=0.13131672212575296)

    term_dictionary = commuting_sets_by_zbasis(term1 + term2)
    true_term_key = ((0, 'Z'), (1, 'X'), (2, 'Z'), (3, 'Y'), (5, 'Y'), (6, 'Z'), (7, 'X'))
    assert list(term_dictionary.keys())[0] == true_term_key


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
    true_set = {((0, 'X'), (1, 'X')): set([x_term.id()]),
                ((0, 'Z'), (1, 'Z')): set([z1_term.id(), z2_term.id(), zz_term.id()])}

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

    true_set = {((1, 'Z'), (2, 'Z'), (3, 'Z'), (4, 'Z')): set(map(lambda x: x.id(), zzzz_terms)),
                ((1, 'X'), (2, 'Z'), (3, 'X'), (4, 'Z')): set(map(lambda x: x.id(), xzxz_terms)),
                ((1, 'X'), (2, 'X'), (3, 'X'), (4, 'X')): set(map(lambda x: x.id(), xxxx_terms)),
                ((1, 'Y'), (2, 'Y'), (3, 'Y'), (4, 'Y')): set(map(lambda x: x.id(), yyyy_terms))}
    for key, value in clumped_terms.items():
        assert set(map(lambda x: x.id(), clumped_terms[key])) == true_set[key]


def test_get_diagonal_basis():
    xxxx_terms = sX(1) * sX(2) + sX(2) + sX(3) * sX(4) + sX(4) + \
                 sX(1) * sX(3) * sX(4) + sX(1) * sX(4) + sX(1) * sX(2) * sX(3)
    true_term = sX(1) * sX(2) * sX(3) * sX(4)
    assert get_diagonalizing_basis(xxxx_terms.terms) == true_term

    zzzz_terms = sZ(1) * sZ(2) + sZ(3) * sZ(4) + \
                 sZ(1) * sZ(3) + sZ(1) * sZ(3) * sZ(4)
    assert get_diagonalizing_basis(zzzz_terms.terms) == sZ(1) * sZ(2) * \
                                                        sZ(3) * sZ(4)


def test_diagonal_basis_commutation():
    x_term = sX(0) * sX(1)
    z1_term = sZ(1)
    z2_term = sZ(0)
    zz_term = sZ(0) * sZ(1)
    assert not diagonal_basis_commutes(x_term, z1_term)
    assert not diagonal_basis_commutes(zz_term, x_term)

    assert diagonal_basis_commutes(z1_term, z2_term)
    assert diagonal_basis_commutes(zz_term, z2_term)
    assert diagonal_basis_commutes(zz_term, z1_term)
    assert diagonal_basis_commutes(zz_term, sI(1))
    assert diagonal_basis_commutes(zz_term, sI(2))
    assert diagonal_basis_commutes(zz_term, sX(5) * sY(7))

