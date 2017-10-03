from grove.alpha.fermion_transforms.fenwick_tree import FenwickTree


def test_sets():
    # tests from 8-qubit example in https://arxiv.org/pdf/1208.5986.pdf
    n_qubits = 8
    tree = FenwickTree(n_qubits)

    parity_dict = {0: set([]),
                   1: set([0]),
                   2: set([1]),
                   3: set([2, 1]),
                   4: set([3]),
                   5: set([4, 3]),
                   6: set([5, 3]),
                   7: set([6, 5, 3])
                  }
    for idx in parity_dict.keys():
        assert set([x.index for x in tree.get_parity_set(idx)]) == \
               parity_dict[idx]

    update_dict = {0: set([1, 3, 7]),
                   1: set([3, 7]),
                   2: set([3, 7]),
                   3: set([7]),
                   4: set([5, 7]),
                   5: set([7]),
                   6: set([7]),
                   7: set([])
                  }
    for idx in update_dict.keys():
        assert set([x.index for x in tree.get_update_set(idx)]) == \
               update_dict[idx]

    flip_dict = {0: set([]),
                 1: set([0]),
                 2: set([]),
                 3: set([2, 1]),
                 4: set([]),
                 5: set([4]),
                 6: set([]),
                 7: set([6, 5, 3])
                }
    for idx in flip_dict.keys():
        assert set([x.index for x in tree.get_children_set(idx)]) == \
               flip_dict[idx]
