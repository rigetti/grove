import numpy as np

from grove.arbitrary_state.unitary_operator import unitary_operator, \
    get_bits_needed


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
