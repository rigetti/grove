from grove.fermion_transforms.fenwick_tree import FenwickTree

def test_sets():
    # tests from 8-qubit example in https://arxiv.org/pdf/1208.5986.pdf
    n_qubits = 8
    tree = FenwickTree(n_qubits)

    parity_dict = {0: {[]},
                   1: {[0]},
                   2: {[1]},
                   3: {[2, 1]},
                   4: {[3]},
                   5: {[4, 3]},
                   6: {[5, 3]},
                   7: {[6, 5, 3]},
                  }
    for idx in parity_dict.keys():
        assert set([x.index for x in tree.get_parity_set(idx)]) == \
               parity_dict[idx]

    update_dict = {0: {[1, 3, 7]},
                   1: {[3, 7]},
                   2: {[3, 7]},
                   3: {[7]},
                   4: {[5, 7]},
                   5: {[7]},
                   6: {[7]},
                   7: {[]},
                  }
    for idx in update_dict.keys():
        assert set([x.index for x in tree.get_update_set(idx)]) == \
               update_dict[idx]

    flip_dict = {0: {[]},
                 1: {[0]},
                 2: {[]},
                 3: {[2, 1]},
                 4: {[]},
                 5: {[4]},
                 6: {[]},
                 7: {[6, 5, 3]},
                }
    for idx in flip_dict.keys():
        assert set([x.index for x in tree.get_children_set(idx)]) == \
               flip_dict[idx]
