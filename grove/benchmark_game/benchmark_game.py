"""A version of a quantum benchmark game, written for PyQuil.

Inspired by https://github.com/decodoku/A_Game_to_Benchmark_Quantum_Computers,
which in turn was inspired by arXiv/1608.00263.
"""
from abc import ABCMeta, abstractmethod

import numpy as np
from networkx import Graph, convert_node_labels_to_integers
from networkx.algorithms.matching import max_weight_matching
from pyquil.api import JobConnection, SyncConnection
from pyquil.gates import CNOT, RY
from pyquil.quil import Program
from six import integer_types


class AbstractBenchmarkGame(object):
    __metaclass__ = ABCMeta

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
        self.G = convert_node_labels_to_integers(G, ordering="sorted")
        self.one_probs_dict = {node: 0 for node in G}
        self.hidden = {node: False for node in G}
        self.prog = Program()
        self.cxn = cxn or SyncConnection()
        self.shots = shots
        self.pair_layers = []
        self.rotation_layers = []
        self.rounds = 0

    @property
    def fuzz(self):
        """
        Calculate the amount of "fuzz" present in the current puzzle.
        Updates self.one_probs_dict as needed.

        :return: The expression

             .. math::

                 sum_{j=0}^n 2\\times \\frac{0.5-\\lvert P_1(j)-0.5\\rvert}{n}

            where :math:`P_1(j)` is the probability of finding qubit :math:`j`
            in the excited state.
        :rtype: float
        """
        self._update_one_probs()
        return 2. / len(self.one_probs_dict) * \
               sum([0.5 - abs(prob - 0.5) for prob
                    in self.one_probs_dict.values()])

    def _generate_disjoint_pairs(self):
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

    def _update_one_probs(self):
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
        n = len(qubits)
        if self.shots is None:
            wvf, _ = self.cxn.wavefunction(self.prog)
            outcome_probs = wvf.get_outcome_probs()
            for bitstring, prob in outcome_probs.items():
                if prob == 0:
                    continue
                for idx, outcome in enumerate(bitstring):
                    if outcome == '1':
                        one_probs_dict[qubits[n - 1 - idx]] += prob
        else:
            if not isinstance(self.shots, integer_types):
                raise TypeError("Shots must be an integer.")
            if self.shots <= 0:
                raise TypeError("Shots must be a positve integer.")

            for q in qubits:
                res = self.cxn.run_and_measure(self.prog, [q], self.shots)
                one_probs_dict[q] = 1.0 * sum([m[0] for m in res]) / self.shots

        self.one_probs_dict = one_probs_dict

    def advance_round(self):
        pairs = self._generate_disjoint_pairs()
        self.pair_layers.append((pairs,))
        self.hidden = {node: False for node in self.G}
        rotations_fracs = []
        for i, j in pairs:
            self.prog.inst(CNOT(i, j))
            frac = np.random.rand()
            rotations_fracs.append(frac)
            self.prog.inst(RY(frac * np.pi, i))
            self.prog.inst(CNOT(i, j))
        self.rotation_layers.append((rotations_fracs,))
        self.rounds += 1

    def choose_pair(self, pair, frac=None):
        if (not isinstance(pair, tuple)) or (len(pair) != 2):
            return False
        i, j = pair
        if self.hidden.get(i, True) or self.hidden.get(j, True):
            return False
        if i not in self.G[j]:
            return False
        self.hidden[i] = True
        self.hidden[j] = True
        if frac is None:
            approximate_one_prob = (self.one_probs_dict[i]
                                    + self.one_probs_dict[j]) / 2.0
            frac = np.arcsin(np.sqrt(approximate_one_prob)) * 2 / np.pi

        self.prog.inst(CNOT(i, j))
        self.prog.inst(RY(-frac * np.pi, i))
        self.prog.inst(CNOT(i, j))

        return True

    def run(self, upper_fuzz_bound=0.9):
        while True:
            x = raw_input("Select how many shots to take: ")
            if len(x) > 0:
                try:
                    x = int(x)
                    break
                except ValueError:
                    continue

            x = None
            break
        self.shots = x
        self.advance_round()
        fuzz = self.fuzz
        print "Round ", self.rounds
        print "Current fuzz: ", fuzz
        target_num_edges = len(self.G) / 2
        edges_chosen = 0
        while True:
            print self
            while True:
                pair_label = raw_input("Choose a pair > ")
                pair = self.get_pair(pair_label)
                valid_pair = self.choose_pair(pair)
                if not valid_pair:
                    print "Invalid pair!"
                else:
                    edges_chosen += 1
                    break
            if edges_chosen == target_num_edges:
                edges_chosen = 0
                fuzz = self.fuzz
                print "You got the fuzz down to: ", fuzz
                if fuzz < 1.0e-5:
                    print "You got rid of the fuzz! Rounds played: ", \
                        self.rounds
                    break
                self.advance_round()
                fuzz = self.fuzz
                print "Round ", self.rounds
                print "Current fuzz: ", fuzz
                if fuzz >= upper_fuzz_bound:
                    print "Too much fuzz: game Over! Rounds played: ", \
                        self.rounds - 1
                    break

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_pair(self, pair_label):
        """
        :param pair_label: Some edge label that determined uniquely a pair
                           of nodes/qubits.
        :return: The tuple (i, j) of nodes/qubits connected by the edge
                 labeled by pair_label; it is up to the inheriting class
                 to decide what that mapping is.
        :rtype: tuple
        """
        pass


class PlanarGridBenchmarkGame(AbstractBenchmarkGame):
    def __init__(self, path):
        self.format_str = ""
        G = self.parse(path)
        super(PlanarGridBenchmarkGame, self).__init__(G)
        self.edge_label_to_pair = {}
        for edge in self.G.edges():
            i, j = edge
            label = self.G[i][j]["label"]
            if label not in self.edge_label_to_pair:
                self.edge_label_to_pair[str(label)] = edge

    def parse(self, path):
        with open(path) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        if len(lines) % 2 == 0:
            raise ValueError("There must be an odd number of lines!")
        puzzle_width = (max([len(line) for line in lines]) + 1) / 2
        G = Graph()
        nodes_found = 0
        edges_found = 0
        node_width = 4
        edge_format = "---()---"
        max_edge_label_size = 2  # assume at most 99 edges --> 2 char labels
        edge_width = len(edge_format) + max_edge_label_size
        for i, line in enumerate(lines):
            if (i % 2) == 0:
                for j, c in enumerate(line):
                    node_number = puzzle_width * i / 2 + j // 2
                    if (j % 2) == 0 and c == "*":
                        G.add_node(node_number)
                        self.format_str += "{" + str(nodes_found) + ":^" \
                                           + str(node_width) + "}"
                        nodes_found += 1
                    elif (j % 2) == 1 and c == "-":
                        G.add_edge(node_number,
                                   node_number + 1,
                                   label=edges_found)
                        self.format_str += \
                            "({0})".format(G[node_number]
                                           [node_number + 1]["label"]) \
                                .center(edge_width, "-")
                        edges_found += 1
                    elif c == " ":
                        self.format_str += " " * edge_width
                    else:
                        raise ValueError("Should not have been "
                                         "this character: " + c)
            else:
                rows = [""] * 3
                for j, c in enumerate(line):
                    if (j % 2) == 0 and c == "|":
                        node_number = puzzle_width * (i // 2) + j / 2
                        G.add_edge(node_number,
                                   node_number + puzzle_width,
                                   label=edges_found)
                        rows[0] += \
                            "|".center(max_edge_label_size + 2)
                        rows[1] += \
                            "({0})".format(G[node_number]
                                           [node_number
                                            + puzzle_width]["label"]) \
                                .center(max_edge_label_size + 2)
                        rows[2] += \
                            "|".center(max_edge_label_size + 2)
                        edges_found += 1
                    elif c == " ":
                        if (j % 2) == 0:
                            spaces = max_edge_label_size + 2
                        else:
                            spaces = edge_width
                        rows[0] += \
                            " " * spaces
                        rows[1] += \
                            " " * spaces
                        rows[2] += \
                            " " * spaces
                    else:
                        raise ValueError("Should not have been "
                                         "this character: " + c)
                self.format_str += "\n".join(rows)

            self.format_str += "\n"

        return G

    def __str__(self):
        vertex_labels = ["*" if self.hidden[node]
                         else str(int(100 * self.one_probs_dict[node])) + "%"
                         for node in range(len(self.G))]
        return self.format_str.format(*vertex_labels)

    def get_pair(self, pair_label):
        return self.edge_label_to_pair.get(pair_label, None)


class RingBenchmarkGame(AbstractBenchmarkGame):
    def __init__(self, n):
        if n % 2 != 0 or n < 2:
            raise ValueError("Must have positive even number of qubits")
        G = Graph()
        G.add_cycle(range(n))
        super(RingBenchmarkGame, self).__init__(G)
        self.format_str = ""
        self.edge_label_to_pair = {}
        self.init_helper()

    def init_helper(self):
        n = len(self.G)
        for i in range(n):
            j = (i + 1) % n
            self.edge_label_to_pair[str(i)] = (i, j)
            self.G[i][j]["label"] = i
        height = n / 4
        width = (n - 2 * height) / 2

        node_width = 4
        label_width = int(np.ceil(np.log10(n))) + 2
        edge_str = "---()---"
        edge_width = node_width + len(edge_str)
        # layer one
        # manually put in first node
        self.format_str += "{0:^" + str(node_width) + "}"
        # set rest of row
        for i in range(width):
            self.format_str += \
                "({0})".format(self.G[i][i + 1]["label"]) \
                    .center(edge_width, "-")
            self.format_str += "{" + str(i + 1) + ":^" + str(node_width) + "}"

        self.format_str += "\n"
        vertical_separators = "|".center(node_width)
        horizontal_spaces_separators = \
            (width + 1) * node_width \
            + edge_width * width \
            - 2 * len(vertical_separators)
        horizontal_spaces_labels = \
            (width + 1) * node_width \
            + edge_width * width \
            - 2 * label_width - 1
        intermediate_rows = vertical_separators + \
                            " " * horizontal_spaces_separators + \
                            vertical_separators + "\n"

        il = (-1 % n)
        jl = 0
        ir = width + 1
        jr = width

        self.format_str += intermediate_rows
        self.format_str += \
            "({0})".format(self.G[il][jl]["label"]) \
                .ljust(label_width) + \
            " " * horizontal_spaces_labels + \
            "({0})".format(self.G[ir][jr]["label"]) \
                .rjust(label_width) + \
            "\n"
        self.format_str += intermediate_rows
        for j in range(height - 1):
            il = (-2 - j) % n
            jl = (-1 - j) % n
            ir = width + 2 + j
            jr = width + 1 + j

            self.format_str += \
                "{" + str(jl) + ":^" + str(node_width) + "}" + \
                " " * horizontal_spaces_labels + \
                "{" + str(jr) + ":^" + str(node_width) + "}" \
                + "\n"

            self.format_str += intermediate_rows
            self.format_str += \
                "({0})".format(self.G[il][jl]["label"]) \
                    .ljust(label_width) + \
                " " * horizontal_spaces_labels + \
                "({0})".format(self.G[ir][jr]["label"]) \
                    .rjust(label_width) + \
                "\n"
            self.format_str += intermediate_rows

        # layer the last
        # manually put in first node
        self.format_str += "{" + str((-height) % n) + ":^" + str(
            node_width) + "}"
        # set rest of row
        for i in range(width):
            il = (-height - i) % n
            jl = il - 1
            self.format_str += \
                "({0})".format(self.G[il][jl]["label"]) \
                    .center(edge_width, "-")
            self.format_str += "{" + str(jl) + ":^" + str(node_width) + "}"

    def __str__(self):
        vertex_labels = ["*" if self.hidden[node]
                         else str(int(100 * self.one_probs_dict[node])) + "%"
                         for node in range(len(self.G))]
        return self.format_str.format(*vertex_labels)

    def get_pair(self, pair_label):
        return self.edge_label_to_pair.get(pair_label, None)
