"""A version of a quantum benchmark game, written for PyQuil.

Inspired by https://github.com/decodoku/A_Game_to_Benchmark_Quantum_Computers,
which in turn was inspired by arXiv/1608.00263.
"""
import numpy as np
from networkx import Graph
from networkx.algorithms.matching import max_weight_matching
from pyquil.api import JobConnection, SyncConnection
from pyquil.quil import Program
from six import integer_types


class AbstractBenchmarkGame(object):
    def __init__(self, G, cxn=None, shots=None):
        """
        Initialize a Game object with a particular graph architecture.

        :param Graph G: the graph to initialize with, must have an even number
                        of nodes.
        :param JobConnection cxn: the connection to run the simulations on.
                                  If None is given, a SyncConnection is used.
        :param int shots: the number of shots to collect for each qubit during
                          the one_prob updating phase.
                          The default, None, means the wavefunction is used
                          directly and exact probabilities are used.
        """
        if len(G) % 2 != 0:
            raise ValueError("Graph G must have an even number of nodes")
        self.G = G
        self.one_probs = {node: 0 for node in G}
        self.hidden = [False] * len(self.one_probs)
        self.prog = Program()
        self.cxn = cxn or SyncConnection()
        self.shots = shots
        self.pair_layers = []
        self.rotation_layers = []

    @property
    def fuzz(self):
        """
        Calculate the amount of "fuzz" present in the current puzzle.

        :return: The expression

             .. math::

                 sum_{j=0}^n 2\\times \\frac{0.5-\\lvert P_1(j)-0.5\\rvert}{n}

            where :math:`P_1(j)` is the probability of finding qubit :math:`j`
            in the excited state.
        :rtype: float
        """
        return 2. / len(self.one_probs) * sum([0.5 - abs(prob - 0.5)
                                               for prob in
                                               self.one_probs.values()])

    def generate_disjoint_pairs(self):
        """
        Randomly partition nodes into disjoint sets of two connected nodes each.

        :return: A list of two-element tuples, such that each tuple
                 contains two nodes that are connected in self.G, and each node
                 appears exactly once
        :rtype: list
        """
        G_copy = Graph()
        G_copy.add_weighted_edges_from([(i, j, np.random.randint(100))
                                        for i, j in self.G.edges()])

        mate = max_weight_matching(G_copy, maxcardinality=True)
        nodes = set(G_copy.nodes())
        pairs = []
        while nodes:
            node = nodes.pop()
            if node in mate:
                pairs.append((node, mate[node]))
                nodes.remove(mate[node])
        return pairs

    def update_one_probs(self):
        """
        For every qubit, get the probability of measuring
        that qubit in the excited state, given that the program p
        is run on self.cxn. Update self's one_probs attribute appropriately.

        :return: a dictionary that is keyed by the qubit with corresponding
                 value the measured proportion that the qubit was in the
                 excited state
        :rtype: dict
        """
        qubits = list(self.prog.get_qubits())
        one_probs_dict = {q: 0 for q in qubits}

        if self.shots is None:
            wvf, _ = self.cxn.wavefunction(self.prog)
            outcome_probs = wvf.get_outcome_probs()
            for bitstring, prob in outcome_probs.items():
                for idx, outcome in enumerate(bitstring):
                    if outcome == '1':
                        one_probs_dict[qubits[idx]] += prob
        else:
            if not isinstance(self.shots, integer_types):
                raise TypeError("Shots must be an integer.")
            if self.shots <= 0:
                raise TypeError("Shots must be a positve integer.")

            for q in qubits:
                res = self.cxn.run_and_measure(self.prog, [q], self.shots)
                one_probs_dict[q] = 1.0 * sum([m[0] for m in res]) / self.shots

        self.one_probs_dict = one_probs_dict
