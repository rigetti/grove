"""
Various ways to group sets of Pauli Terms

This augments the existing infrastructure in pyquil that finds commuting sets
of PauliTerms.
"""
from functools import reduce
from pyquil.paulis import is_identity, PauliTerm, PauliSum


def _commutes(p1, p2):
    # Identity commutes with anything
    if is_identity(p1) or is_identity(p2):
        return True

    # Operators acting on different qubits commute
    if len(set(p1.get_qubits()) & set(p2.get_qubits())) == 0:
        return True

    # Otherwise, they must be the same thing modulo coefficient
    return p1.id() == p2.id()


def diagonal_basis_commutes(pauli_a, pauli_b):
    """
    Test if `pauli_a` and `pauli_b` share a diagonal basis

    Example:

        Check if [A, B] with the constraint that A & B must share a one-qubit
        diagonalizing basis. If the inputs were [sZ(0), sZ(0) * sZ(1)] then this
        function would return True.  If the inputs were [sX(5), sZ(4)] this
        function would return True.  If the inputs were [sX(0), sY(0) * sZ(2)]
        this function would return False.

    :param pauli_a: Pauli term to check commutation against `pauli_b`
    :param pauli_b: Pauli term to check commutation against `pauli_a`
    :return: Boolean of commutation result
    :rtype: Bool
    """
    overlapping_active_qubits = set(pauli_a.get_qubits()) & set(pauli_b.get_qubits())
    for qubit_index in overlapping_active_qubits:
        if (pauli_a[qubit_index] != 'I' and pauli_b[qubit_index] != 'I' and
           pauli_a[qubit_index] != pauli_b[qubit_index]):
            return False

    return True


def get_diagonalizing_basis(list_of_pauli_terms):
    """
    Find the Pauli Term with the most non-identity terms

    :param list_of_pauli_terms: List of Pauli terms to check
    :return: The highest weight Pauli Term
    :rtype: PauliTerm
    """
    qubit_ops = set(reduce(lambda x, y: x + y,
                       [list(term._ops.items()) for term in list_of_pauli_terms]))
    qubit_ops = sorted(list(qubit_ops), key=lambda x: x[0])

    return PauliTerm.from_list(list(map(lambda x: tuple(reversed(x)), qubit_ops)))


def _max_key_overlap(pauli_term, diagonal_sets):
    """
    Calculate the max overlap of a pauli term ID with keys of diagonal_sets

    Returns a different key if we find any collisions.  If no collisions is
    found then the pauli term is added and the key is updated so it has the
    largest weight.

    :param pauli_term:
    :param diagonal_sets:
    :return: dictionary where key value pair is tuple indicating diagonal basis
             and list of PauliTerms that share that basis
    :rtype: dict
    """
    # a lot of the ugliness comes from the fact that
    # list(PauliTerm._ops.items()) is not the appropriate input for
    # Pauliterm.from_list()
    for key in list(diagonal_sets.keys()):
        pauli_from_key = PauliTerm.from_list(
            list(map(lambda x: tuple(reversed(x)), key)))
        if diagonal_basis_commutes(pauli_term, pauli_from_key):
            updated_pauli_set = diagonal_sets[key] + [pauli_term]
            diagonalizing_term = get_diagonalizing_basis(updated_pauli_set)
            if len(diagonalizing_term) > len(key):
                del diagonal_sets[key]
                new_key = tuple(sorted(diagonalizing_term._ops.items(),
                                       key=lambda x: x[0]))
                diagonal_sets[new_key] = updated_pauli_set
            else:
                diagonal_sets[key] = updated_pauli_set
            return diagonal_sets
    # made it through all keys and sets so need to make a new set
    else:
        # always need to sort because new pauli term functionality
        new_key = tuple(sorted(pauli_term._ops.items(), key=lambda x: x[0]))
        diagonal_sets[new_key] = [pauli_term]
        return diagonal_sets


def commuting_sets_by_zbasis(pauli_sums):
    """
    Computes commuting sets based on terms having the same diagonal basis

    Following the technique outlined in the appendix of arXiv:1704.05018.

    :param pauli_sums: PauliSum object to group
    :return: dictionary where key value pair is a tuple corresponding to the
             basis and a list of PauliTerms associated with that basis.
    """
    diagonal_sets = {}
    for term in pauli_sums:
        diagonal_sets = _max_key_overlap(term, diagonal_sets)

    return diagonal_sets


def check_trivial_commutation(pauli_list, single_pauli_term):
    """
    Check if a PauliTerm trivially commutes with a list of other terms.

    :param list pauli_list: A list of PauliTerm objects
    :param PauliTerm single_pauli_term: A PauliTerm object
    :returns: True if pauli_two object commutes with pauli_list, False otherwise
    :rtype: bool
    """
    if not isinstance(pauli_list, list):
        raise TypeError("pauli_list should be a list")

    for term in pauli_list:
        if not _commutes(term, single_pauli_term):
            return False
    return True


def commuting_sets_by_indices(pauli_sums, commutation_check):
    """
    For a list of pauli sums, find commuting sets and keep track of which pauli sum they came from.

    :param pauli_sums: A list of PauliSum
    :param commutation_check: a function that checks if all elements of a list
                              and a single pauli term commute.
    :return: A list of commuting sets. Each set is a list of tuples (i, j) to find the particular
        commuting term. i is the index of the pauli sum from whence the term came. j is the
        index within the set.
    """
    assert isinstance(pauli_sums, list)

    group_inds = []
    group_terms = []
    for i, pauli_sum in enumerate(pauli_sums):
        for j, term in enumerate(pauli_sum):
            if len(group_inds) == 0:
                # Initialization
                group_inds.append([(i, j)])
                group_terms.append([term])
                continue

            for k, group in enumerate(group_terms):
                if commutation_check(group, term):
                    group_inds[k] += [(i, j)]
                    group_terms[k] += [term]
                    break
            else:
                # for ... else means loop completed without a `break`
                # Which means this needs to start its own group.
                group_inds.append([(i, j)])
                group_terms.append([term])

    return group_inds


def commuting_sets_trivial(pauli_sum):
    """
    Group a pauli term into commuting sets using trivial check

    :param pauli_sum: PauliSum term
    :return: list of lists containing individual Pauli Terms
    """
    if not isinstance(pauli_sum, (PauliTerm, PauliSum)):
        raise TypeError("This method can only group PauliTerm or PauliSum objects")

    if isinstance(pauli_sum, PauliTerm):
        pauli_sum = PauliSum([pauli_sum])

    commuting_terms = []
    for term in pauli_sum:
        # find the group that it trivially commutes with
        for term_group in commuting_terms:
            if check_trivial_commutation(term_group, term):
                term_group.append(term)
                break
        else:
            commuting_terms.append([term])

    return commuting_terms

