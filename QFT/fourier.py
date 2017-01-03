import pyquil.quil as pq
from pyquil.gates import SWAP, H, CPHASE
import math


def bit_reversal(qubits):
    """
    Generate a circuit to do bit reversal.

    :param qubits: Qubits to do bit reversal with.
    :return: A program to do bit reversal.
    """
    p = pq.Program()
    n = len(qubits)
    for i in xrange(n / 2):
        p.inst(SWAP(qubits[i], qubits[-i - 1]))
    return p


def qft(qubits):
    """
    Generate a program to compute the quantum Fourier transform on
    a set of qubits.

    :param qubits: A list of qubit indexes.
    :return: A Quil program to compute the Fourier transform of the qubits.
    """

    def core_qft(qubits):
        q = qubits[0]
        qs = qubits[1:]
        if 1 == len(qubits):
            return [H(q)]
        else:
            n = 1 + len(qs)
            cR = []
            for idx, i in enumerate(xrange(n - 1, 0, -1)):
                q_idx = qs[idx]
                angle = math.pi / 2 ** (n - i)
                cR.append(CPHASE(angle)(q, q_idx))
            return core_qft(qs) + list(reversed(cR)) + [H(q)]

    p = pq.Program().inst(core_qft(qubits))
    return p + bit_reversal(qubits)


if __name__ == '__main__':
    print qft([0, 1, 2, 3])
