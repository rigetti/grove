"""
Various ways to group sets of Pauli Terms

This augments the existing infrastructure in pyquil that finds commuting sets
of PauliTerms.
"""
from pyquil.paulis import check_commutation


def check_trivial_commutation(pauli_list, pauli_two):
    """
    Check if a PauliTerm trivially commutes with a list of other terms.

    :param list pauli_list: A list of PauliTerm objects
    :param PauliTerm pauli_two_term: A PauliTerm object
    :returns: True if pauli_two object commutes with pauli_list, False otherwise
    :rtype: bool
    """
    if not isinstance(pauli_list, list):
        raise TypeError("pauli_list should be a list")

    def _commutes(p1, p2):
        # Identity commutes with anything
        if len(p1.get_qubits()) == 0 or len(p1.get_qubits()) == 0:
            return True

        # Operators acting on different qubits commute
        if len(set(p1.get_qubits()) & set(p2.get_qubits())) == 0:
            return True

        # Otherwise, they must be the same thing modulo coefficient
        return p1.id() == p2.id()

    for i, term in enumerate(pauli_list):
        if not _commutes(term, pauli_two):
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

    :param pauli_sums:
    :return:
    """
    def _max_key_overlap(item, diagonal_sets, max_qubit):
        """
        Calculate the max overlap of item with keys of diagoanl_sets

        Returns a different key if we find any collisions

        This mutates the sets

        :param item:
        :param diagonal_sets:
        :return:
        """
        hash_ptp = tuple([item[n] for n in range(max_qubit)])

        keys = list(diagonal_sets.keys())
        #  if there are keys check for collisions if not return updated
        #  diagonal_set dictionary with the key and term added
        for key in keys:  # for each key check any collisions
            for idx, pauli_tensor_element in enumerate(key):
                if ((pauli_tensor_element != 'I' and hash_ptp[idx] != 'I')
                   and hash_ptp[idx] != pauli_tensor_element):
                    #  item has collision with this key
                    #  so must be in a different in a different key or new key
                    break
            else:
                #  we've gotten to the end without finding a difference!
                #  that means this key works with this item!
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
                    diagonal_sets[tuple(new_key)].append(item)
                else:
                    diagonal_sets[tuple(new_key)] = diagonal_sets[key]
                    diagonal_sets[tuple(new_key)].append(item)
                    del diagonal_sets[key]
                return diagonal_sets

        diagonal_sets[hash_ptp] = [item]
        return diagonal_sets

    max_qubit = 0
    for term in pauli_sums:
        if term.id() != "":
            max_qubit = max(max_qubit, max(term.get_qubits()))
    max_qubit += 1  # index from 1

    diagonal_sets = {}
    for term in pauli_sums:
        diagonal_sets = _max_key_overlap(term, diagonal_sets, max_qubit)

    return diagonal_sets

