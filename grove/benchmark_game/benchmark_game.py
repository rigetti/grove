import numpy as np
from networkx import Graph
from networkx.algorithms.matching import max_weight_matching
from pyquil.api import JobConnection
from pyquil.quil import Program
from six import integer_types


def get_disjoint_pairs(G):
    """
    Randomly partition nodes into disjoint sets of two connected nodes each.

    :param Graph G: An undirected graph with qubits as nodes and
                    edges as connections between qubits that allow for
                    a two qubit gate to be applied.
    :return: A list of two-element tuples, such that each tuple
             contains two nodes that are connected in G, and each node
             appears exactly once
    :rtype: list
    """
    G_copy = Graph()
    G_copy.add_weighted_edges_from([(i, j, np.random.randint(100))
                                    for i, j in G.edges()])

    mate = max_weight_matching(G_copy, maxcardinality=True)
    nodes = set(G_copy.nodes())
    pairs = []
    while nodes:
        node = nodes.pop()
        if node in mate:
            pairs.append((node, mate[node]))
            nodes.remove(mate[node])
    return pairs


def calculate_fuzz(one_probs):
    """
    Calculate the amount of "fuzz" present in the current puzzle.

    :param list one_probs: A list of probabilities corresponding to the
                           probability of measuring each qubit in the
                           excited state.
    :return: The expression

             .. math::

                 sum_{j=0}^n 2\\times \\frac{0.5-\\lvert P_1(j)-0.5\\rvert}{n}
    :rtype: float
    """
    return 2. / len(one_probs) * sum([0.5 - abs(prob - 0.5)
                                      for prob in one_probs])


def get_one_probs(p, cxn, shots=None):
    """
    For every qubit, get the probability of measuring
    that qubit in the excited state, given that the program p
    is run.

    :param Program p: the program to run on the ground state.
    :param JobConnection cxn: the connection to run the simulations on.
    :param int shots: the number of shots to collect for each qubit.
                      The default, None, means to use the wavefunction
                      directly and get the exact results.
    :return: a dictionary that is keyed by the qubit with corresponding
             value the measured proportion that the qubit was in the
             excited state
    :rtype: list
    """
    one_probs = {}
    qubits = p.get_qubits()

    if shots is None:
        wf, _ = cxn.wavefunction(p)
        for i, q in enumerate(qubits):
            one_probs[q] = np.abs(wf[i]) ** 2

    if not isinstance(shots, integer_types):
        raise TypeError("Shots must be an integer.")
    if shots <= 0:
        raise TypeError("Shots must be a positve integer.")
    for q in qubits:
        res = cxn.run_and_measure(p, [q], shots)
        one_probs[q] = 1.0 * sum([m[0] for m in res]) / shots
    return one_probs
