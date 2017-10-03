import numpy as np

from grove.alpha.arbitrary_state.unitary_operator import unitary_operator, \
    get_bits_needed, fix_norm_and_length


def test_unitary():
    length = 30
    # make an arbitrary complex vector
    v = np.random.uniform(-1, 1, length) \
        + np.random.uniform(-1, 1, length) * 1j

    # normalize and pad with zeros
    v_norm = v / np.linalg.norm(v)
    num_bits = get_bits_needed(length)
    while len(v_norm) < 2 ** num_bits:
        v_norm = np.append(v_norm, 0)

    # generate unitary operator
    U = unitary_operator(v_norm)

    # make sure U|0> = |v>
    zero_state = np.zeros(len(U))
    zero_state[0] = 1
    assert np.allclose(U.dot(zero_state), v_norm)


def test_fix_norm_and_length():
    length = 30
    # make an arbitrary complex vector
    v = np.random.uniform(-1, 1, length) \
        + np.random.uniform(-1, 1, length) * 1j

    new_v = fix_norm_and_length(v)
    assert np.allclose([np.linalg.norm(new_v)], [1])
    assert len(new_v) == 2 ** get_bits_needed(len(v))


def test_get_bits_needed():
    for i, j in zip([1, 2, 5, 8, 9, 14, 27], [1, 1, 3, 3, 4, 4, 5]):
        assert get_bits_needed(i) == j
