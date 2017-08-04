#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
Class to represent a Fenwick tree.
"""


class FenwickNode:
    """Fenwick Tree node."""
    parent = None
    children = None
    index = None

    def __init__(self, parent, children, index=None):
        """Fenwick Tree node. Single parent and multiple children.
        
        :param FenwickNode parent: a parent node
        :param list(FenwickNode) children: a list of children nodes
        :param int index: node label
        """
        self.children = children
        self.parent = parent
        self.index = index

    def get_ancestors(self):
        """Returns a list of ancestors of the node. Ordered from the earliest.

        :return: node's ancestors, ordered from most recent
        :rtype: list(FenwickNode)
        """
        node = self
        ancestor_list = []
        while node.parent is not None:
            ancestor_list.append(node.parent)
            node = node.parent

        return ancestor_list


class FenwickTree:
    """Recursive implementation of the Fenwick tree.
    Please see Subsection B.2. of Operator Locality in Quantum
    Simulation of Fermionic Models (arXiv:1701.07072) for
    a reference to the update set (U), the parity set (P) and the
    children set (F) sets of the Fenwick.
    """
    # Root node.
    root = None

    def __init__(self, n_qubits):
        """
        Builds a Fenwick tree on n_qubits qubits.

        :param int n_qubits: number of qubits in the system
        """
        self.nodes = [FenwickNode(None, []) for _ in range(n_qubits)]

        if n_qubits > 0:
            self.root = self.nodes[n_qubits - 1]
            self.root.index = n_qubits - 1

        def fenwick(left, right, parent):
            """
            This inner function is used to build the Fenwick tree on nodes
            recursively. See Algorithm 1 in the paper.

            :param int left: left boundary of range
            :param int right: right boundary of range
            :param FenwickNode parent: parent node
            """
            if left >= right:
                return
            else:
                pivot = (left + right) >> 1
                child = self.nodes[pivot]

                # The circle of life:
                # Parent has child.
                # Child becomes parent.
                child.index = pivot
                parent.children.append(child)
                child.parent = parent

                # Recurse to left and to right.
                fenwick(left, pivot, child)
                fenwick(pivot + 1, right, parent)

        # Builds structure on nodes.
        fenwick(0, n_qubits - 1, self.root)

    def get_node(self, j):
        """Returns the node at j in the qubit register. Wrapper.

        :param int j: fermionic site index

        :return: the node at j
        :rtype: FenwickNode
        """
        return self.nodes[j]

    def get_update_set(self, j):
        """The set of all ancestors of j, (the update set U from the paper).

        :param int j: fermionic site index

        :return: ancestors of j, ordered from most recent
        :rtype: list(FenwickNode)
        """
        node = self.get_node(j)
        return node.get_ancestors()

    def get_children_set(self, j):
        """Returns the set of children of j-th site.

        :param int j: fermionic site index

        :return: children of j, ordered from lowest index
        :rtype: list(FenwickNode)
        """
        node = self.get_node(j)
        return node.children

    def get_remainder_set(self, j):
        """Return the set of children with indices less than j of all ancestors
        of j. The set C from (arXiv:1701.07072).

        :param int j: fermionic site index

        :return: children of j-ancestors, with indices less than j
        :rtype: list(FenwickNode)
        """
        result = []
        ancestors = self.get_update_set(j)

        # This runs in O(log(N)log(N)) where N is the number of qubits.
        for a in ancestors:
            for c in a.children:
                if c.index < j:
                    result.append(c)

        return result

    def get_parity_set(self, j):
        """Returns the union of the remainder set with children set. Coincides
        with the parity set of Tranter et al.

        :param int j: fermionic site index

        :return: union of remainder and parity set for node with index j
        :rtype: list(FenwickNode)
        """
        return self.get_remainder_set(j) + self.get_children_set(j)
