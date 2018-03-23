"""
Various ways to group sets of Pauli Terms

This augments the existing infrastructure in pyquil that finds commuting sets
of PauliTerms.
"""
from pyquil.paulis import check_commutation, is_identity, PauliTerm, PauliSum


def _commutes(p1, p2):
    # Identity commutes with anything
    if is_identity(p1) or is_identity(p2):
        return True

    # Operators acting on different qubits commute
    if len(set(p1.get_qubits()) & set(p2.get_qubits())) == 0:
        return True

    # Otherwise, they must be the same thing modulo coefficient
    return p1.id() == p2.id()


def _max_key_overlap(pauli_term, diagonal_sets, active_qubits):
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
    hash_ptp = tuple([pauli_term[n] for n in active_qubits])

    keys = list(diagonal_sets.keys())
    #  if there are keys check for collisions if not return updated
    #  diagonal_set dictionary with the key and term added
    for key in keys:  # for each key check any collisions
        for idx, pauli_tensor_element in enumerate(key):
            if ((pauli_tensor_element != 'I' and hash_ptp[idx] != 'I')
               and hash_ptp[idx] != pauli_tensor_element):
                #  item has collision with this key
                #  so must be  a different key or new key
                break
        else:
            #  we've gotten to the end without finding a difference!
            #  that means this key works with this pauli term!
            #  Now we must select the longer of the two keys
            #  longer is the key or item with fewer identities
            new_key = []
            for ii in range(len(hash_ptp)):
                if hash_ptp[ii] != 'I':
                    new_key.append(hash_ptp[ii])
                elif key[ii] != 'I':
                    new_key.append(key[ii])
                else:
                    new_key.append('I')

            if tuple(new_key) in diagonal_sets.keys():
                diagonal_sets[tuple(new_key)].append(pauli_term)
            else:
                diagonal_sets[tuple(new_key)] = diagonal_sets[key]
                diagonal_sets[tuple(new_key)].append(pauli_term)
                del diagonal_sets[key]
            return diagonal_sets

    diagonal_sets[hash_ptp] = [pauli_term]
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


def commuting_sets_by_zbasis(pauli_sums):
    """
    Computes commuting sets based on terms having the same diagonal basis

    Following the technique outlined in the appendix of arXiv:1704.05018.

    :param pauli_sums: PauliSum object to group
    :return: dictionary where key value pair is a tuple corresponding to the
             basis and a list of PauliTerms associated with that basis.
    """
    active_qubits = []
    for term in pauli_sums:
        active_qubits += list(term.get_qubits())
    # get unique indices and put in order from least to greatest
    # NOTE: translation layer to physical qubits is likely to be needed
    active_qubits = sorted(list(set(active_qubits)))

    diagonal_sets = {}
    for term in pauli_sums:
        diagonal_sets = _max_key_overlap(term, diagonal_sets, active_qubits)

    return diagonal_sets


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

