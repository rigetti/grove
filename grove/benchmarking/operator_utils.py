"""
Utilities for encapsulating bases and properties of quantum operators and super-operators
as represented by qutip.Qobj()'s.
"""

from collections import OrderedDict, namedtuple

import itertools

import logging
import numpy as np
import qutip as qt
from scipy.sparse.linalg import norm as spnorm
from scipy.sparse import hstack as sphstack, vstack as spvstack
import pyquil as pq
import matplotlib.pyplot as plt

_log = logging.getLogger(__name__)

# PAULI OPS
X = qt.sigmax()
Y = qt.sigmay()
Z = qt.sigmaz()
I = qt.qeye(2)
TOMOGRAPHY_GATES = {
    "X-HALF": (-1j * np.pi / 4 * X).expm(),
    "Y-HALF": (-1j * np.pi / 4 * Y).expm(),
    "X": (-1j * np.pi / 2 * X).expm(),
    "Y": (-1j * np.pi / 2 * Y).expm(),
    "MINUS-X-HALF": (1j * np.pi / 4 * X).expm(),
    "MINUS-Y-HALF": (1j * np.pi / 4 * Y).expm(),
    "MINUS-X": (1j * np.pi / 2 * X).expm(),
    "MINUS-Y": (1j * np.pi / 2 * Y).expm(),
    "I": qt.qeye(2)
}

EPS = 1e-8


def quil_to_operator(program):
    """
    Convert a proto-Quil program consisting only of single qubit gates that appear in
    ``TOMOGRAPHY-GATES`` to a ``qutip.Qobj operator``.

    :param pyquil.quil.Program program: The program to convert.
    :return: A unitary operator representation.
    :rtype: qutip.Qobj
    """

    qubits = sorted(q for q in program.extract_qubits())
    U = qt.qeye([2] * len(qubits))

    for g in program.synthesize():

        if not isinstance(g, pq.gates.Gate):
            raise ValueError("Unsupported Instruction: %s" % g)

        if not len(g.arguments) == 1:
            raise ValueError("Only programs comprised of single qubit gates supported: %s" % g)

        qt_op = TOMOGRAPHY_GATES[g.operator_name]

        ops = [I] * len(qubits)
        idx = qubits.index(g.arguments[0].index())
        ops[idx] = qt_op

        U = qt.tensor(*ops) * U
    return U


def state_histogram(state, ax, title):
    """
    Visualize a quantum state in some specific axes instance and set a title.

    :param qutip.Qobj state: The quantum state.
    :param matplotlib.Axes ax: The matplotlib axes.
    :param str title: The title for the plot.
    """

    qt.matrix_histogram_complex(state, limits=[-1, 1], ax=ax)
    ax.view_init(azim=-55, elev=45)
    ax.set_title(title)


def plot_pauli_rep(state_coeffs, ax, labels, title):
    """
    Visualize a state in a generalized Pauli-Basis as a bar-plot.

    :param numpy.ndarray state_coeffs: The state represented in the generalized basis.
    If the generalized basis is hermitian, the state_coeffs should be purely real. This function
    therefore only visualizes the real part.
    :param ax: The matplotlib axes.
    :param labels: The labels for the operator basis states.
    :param title: The title for the plot
    """
    dim = len(labels)
    im = ax.bar(np.arange(dim) - .4, np.real(state_coeffs), width=.8)
    ax.set_xticks(xrange(dim))
    ax.set_xlabel("Pauli Operator")
    ax.set_ylabel("Coefficient")
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(False)


def plot_pauli_transfer_matrix(ptransfermatrix, ax, labels, title):
    """
    Visualize the Pauli Transfer Matrix of a process.

    :param numpy.ndarray ptransfermatrix: The Pauli Transfer Matrix
    :param ax: The matplotlib axes.
    :param labels: The labels for the operator basis states.
    :param title: The title for the plot
    """
    im = ax.imshow(ptransfermatrix, interpolation="nearest", cmap="RdBu", vmin=-1, vmax=1)
    dim = len(labels)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(xrange(dim))
    ax.set_xlabel("Input Pauli Operator")
    ax.set_yticks(xrange(dim))
    ax.set_ylabel("Output Pauli Operator")
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.grid(False)


class CRMBaseError(Exception):
    """
    Base class for errors raised when the confusion rate matrix is defective.
    """
    pass


class CRMUnnormalizedError(CRMBaseError):
    """
    Raised when a confusion rate matrix is not properly normalized.
    """
    pass


class CRMValueError(CRMBaseError):
    """
    Raised when a confusion rate matrix contains elements not contained in the interval :math`[0,1]`
    """
    pass


DiagonalPOVM = namedtuple("DiagonalPOVM", ["pi_basis", "confusion_rate_matrix", "ops"])


def make_diagonal_povm(pi_basis, confusion_rate_matrix):
    """
    Create a DiagonalPOVM from a ``pi_basis`` and the ``confusion_rate_matrix`` associated with a
    readout.

    See also the willow docs on :ref:`measurement-docs`.

    :param OperatorBasis pi_basis: An operator basis of rank-1 projection operators.
    :param numpy.ndarray confusion_rate_matrix: The matrix of detection probabilities conditional
    on a prepared qubit state.
    """

    confusion_rate_matrix = np.asarray(confusion_rate_matrix)
    if not np.allclose(confusion_rate_matrix.sum(axis=0), np.ones(confusion_rate_matrix.shape[1])):
        raise CRMUnnormalizedError("Unnormalized confusion matrix:\n{}".format(
            confusion_rate_matrix))
    if not (confusion_rate_matrix >= 0).all() or not (confusion_rate_matrix <= 1).all():
        raise CRMValueError("Confusion matrix must have values in [0, 1]:"
                                          "\n{}".format(confusion_rate_matrix))

    ops = [sum((pi_j * pjk for (pi_j, pjk) in itertools.izip(pi_basis.ops, pjs)), 0)
                for pjs in confusion_rate_matrix]
    return DiagonalPOVM(pi_basis=pi_basis, confusion_rate_matrix=confusion_rate_matrix, ops=ops)


class OperatorBasis(object):
    """
    Encapsulate a complete set of basis operators.
    """

    def __init__(self, labels_ops):
        """
        Encapsulates a set of linearly independent operators.

        :param (list|tuple) labels_ops: Sequence of tuples (label, operator) where label is a string
            and operator a qutip.Qobj operator representation.
        """
        self.ops_by_label = OrderedDict(labels_ops)
        self.labels = list(self.ops_by_label.keys())
        self.ops = list(self.ops_by_label.values())
        self.dim = len(self.ops)

        # the basis change transformation matrix from a representation in the operator basis
        # to the original basis. We enforce CSR sparse matrix representation to have efficient
        # matrix vector products.
        self.basis_transform = sphstack([qt.operator_to_vector(opj).data
                                         for opj in self.ops]).tocsr()

        self._metric = None
        self._is_orthonormal = None
        self._all_hermitian = None

    def metric(self):
        """
        Compute a matrix of Hilbert-Schmidt inner products for the basis operators.
        """
        if self._metric is None:
            _log.debug("Computing and caching operator basis metric")
            self._metric = np.matrix([[(j.dag() * k).tr() for k in self.ops] for j in self.ops])
        return self._metric

    def is_orthonormal(self):
        """
        Compute a matrix of Hilbert-Schmidt inner products for the basis operators.
        """
        if self._is_orthonormal is None:
            _log.debug("Testing and caching if operator basis is orthonormal")
            self._is_orthonormal = np.allclose(self.metric(), np.eye(self.dim))
        return self._is_orthonormal

    def all_hermitian(self):
        """
        Check if all basis operators are hermitian.
        """
        if self._all_hermitian is None:
            _log.debug("Testing and caching if all basis operator are hermitian")
            self._all_hermitian = all((is_hermitian(op) for op in self.ops))
        return self._all_hermitian

    def __iter__(self):
        """
        Iterate over tuples of (label, basis_op)
        """
        for l, op in zip(self.labels, self.ops):
            yield l, op

    def product(self, *bases):
        """
        Compute the tensor product with another basis.

        :param bases: One or more additional bases to form the product with.
        :return (OperatorBasis): The tensor product basis as an OperatorBasis object.
        """
        if len(bases) > 1:
            basis_rest = bases[0].product(*bases[1:])
        else:
            assert len(bases) == 1
            basis_rest = bases[0]

        labels_ops = [(b1l + b2l, qt.tensor(b1, b2)) for (b1l, b1), (b2l, b2) in
                      itertools.product(self, basis_rest)]

        return OperatorBasis(labels_ops)

    def __pow__(self, n):
        """
        Create the n-fold tensor product basis.

        :param (int) n: The number of identical tensor factors.
        :return (OperatorBasis): The product basis.
        """
        if not isinstance(n, int):  # pragma no coverage
            raise TypeError("Can only accept an integer number of factors")
        if n < 1:  # pragma no coverage
            raise ValueError("Need positive number of factors")
        if n == 1:
            return self

        return self.product(*([self] * (n - 1)))

    def super_basis(self):
        """
        Generate the superoperator basis in which the Choi matrix can be represented.

        The follows the definition in [Chow]_

        .. [Chow] Chow et al., 2012, https://doi.org/10.1103/PhysRevLett.109.060501

        :return (OperatorBasis): The super basis as an OperatorBasis object.
        """

        labels_ops = [(bnl + "^T (x) " + bml, qt.sprepost(bm, bn)) for (bnl, bn), (bml, bm) in
                      itertools.product(self, self)]
        return OperatorBasis(labels_ops)

    def project_op(self, op):
        """
        Project an operator onto the basis.

        :param qutip.Qobj op: The operator to project.
        :return: The projection coefficients as a numpy array.
        :rtype: scipy.sparse.csr_matrix
        """
        if not self.is_orthonormal():  # pragma no coverage
            raise ValueError("project_op only implemented for orthonormal operator bases")
        return self.basis_transform.H * qt.operator_to_vector(op).data

    def transfer_matrix(self, sop):
        """
        Compute the transfer matrix :math:`R_{jk} = \tr[P_j sop(P_k)]`.

        :param qutip.Qobj sop: The superoperator to transform.
        :return: The transfer matrix in dense or sparse form.
        :rtype: scipy.sparse.csr_matrix
        """
        if not self.is_orthonormal():  # pragma no coverage
            raise ValueError("transfer_matrix() only implemented for orthonormal operator bases")
        return self.basis_transform.H * sop.data * self.basis_transform

    def super_from_tm(self, tm):
        """
        Reconstruct a super operator from a transfer matrix representation.
        This inverts `self.transfer_matrix(...)`.

        :param (numpy.ndarray) tm: A process in transfer matrix form.
        :return: A qutip super operator.
        :rtype: qutip.Qobj.
        """
        if not self.is_orthonormal():  # pragma no coverage
            raise ValueError("super_from_tm() only implemented for orthonormal operator bases")

        data = self.basis_transform * tm * self.basis_transform.H
        sop = qt.Qobj(data, dims=[self.ops[0].dims, self.ops[0].dims])
        sop.superrep = "super"
        return sop

    def __str__(self):
        return "<span[{}]>".format(",".join(self.labels))

    def __eq__(self, other):
        return (self.labels == other.labels and all(
            [(my_op - o_op).norm('fro') < EPS for (my_op, o_op) in zip(self.ops, other.ops)]))


PAULI_BASIS = OperatorBasis(
    [("I", I / np.sqrt(2)), ("X", X / np.sqrt(2)), ("Y", Y / np.sqrt(2)), ("Z", Z / np.sqrt(2))])


def gell_mann_basis(qudit_dim):
    """
    Compute an orthonormal hermitian operator basis for the space of observables of a qudit with
    `d` levels. These are identical to the Gell-Mann matrices up to normalization.
    See also this
    `wikipedia article <https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices>`_ .

    :param int qudit_dim: The dimension of the underlying Hilbert space.
    :return: The normalized Gell-Mann operator basis.
    :rtype: OperatorBasis
    """

    labels_ops = [('I', qt.qeye(qudit_dim) / np.sqrt(qudit_dim))]
    xops = []
    yops = []

    # off diagonals
    for jj in range(qudit_dim):
        for kk in range(jj):
            sigma_jk = qt.projection(qudit_dim, jj, kk)
            xops.append(('X{}{}'.format(kk, jj), (sigma_jk + sigma_jk.dag()) / np.sqrt(2)))
            yops.append(('Y{}{}'.format(kk, jj), (sigma_jk - sigma_jk.dag()) / np.sqrt(2) * 1j))

    labels_ops += xops + yops

    # diagonals
    for jj in range(1, qudit_dim):
        diag = np.array([1.] * jj + [-jj] + [0.] * (qudit_dim - jj - 1))
        diag /= np.linalg.norm(diag, 2)
        labels_ops.append(('Z{}'.format(jj),
                           qt.qdiags([diag], [0], [[qudit_dim], [qudit_dim]],
                                     (qudit_dim, qudit_dim))))

    return OperatorBasis(labels_ops)


def n_qubit_pauli_basis(n):
    """
    Construct the tensor product operator basis of `n` PAULI_BASIS's.

    :param int n: The number of qubits.
    :return OperatorBasis: The product Pauli operator basis of `n` qubits
    """
    if n >= 1:
        return PAULI_BASIS ** n
    else:  # pragma no coverage
        raise ValueError("n = {} should be at least 1.".format(n))


def n_qubit_ground_state(n):
    """
    Construct the tensor product of `n` ground states |0>.

    :param int n: The number of qubits.
    :return qutip.Qobj: The state |000...0> for `n` qubits.
    """
    return qt.tensor(*([GS] * n))


def to_density_matrix(state):
    """
    Convert a Hilbert space vector to a density matrix.
    """
    return state * state.dag()


GS = to_density_matrix(qt.basis(2, 0))
ES = to_density_matrix(qt.basis(2, 1))


def generated_states(initial_state, preparations):
    """
    Generate states prepared from channel operators acting on an initial state.
    Typically the channel operators will be unitary.

    :param qutip.Qobj initial_state: The initial state as a density matrix.
    :param (list|tuple) preparations: The unitary channel operators that transform the initial state.
    """
    return [e * initial_state * e.dag() for e in preparations]


def is_hermitian(operator):
    """
    Check if matrix or operator is hermitian.

    :param (numpy.ndarray|qutip.Qobj) operator: The operator or matrix to be tested.
    :return: True if the operator is hermitian.
    :rtype: bool
    """
    if isinstance(operator, qt.Qobj):
        return (operator.dag() - operator).norm('fro') / operator.norm('fro') < EPS
    if isinstance(operator, np.ndarray):
        return np.linalg.norm(operator.T.conj() - operator) / np.linalg.norm(operator) < EPS
    return spnorm(operator.H - operator) / spnorm(operator) < EPS


def is_projector(operator):
    """
    Check if operator is a projector.

    :param qutip.Qobj operator: The operator or matrix to be tested.
    :return: True if the operator is a projector.
    :rtype: bool
    """
    # verify that P^dag=P and P^2-P=0 holds up to relative numerical accuracy EPS.
    return (is_hermitian(operator) and (operator * operator - operator).norm('fro') / operator.norm(
        'fro') < EPS)


def choi_matrix(pauli_tm, basis):
    """
    Compute the Choi matrix for a quantum process from its Pauli Transfer Matrix.

    This agrees with the definition in [Chow]_ except for a different overall normalization.
    Our normalization agrees with that of qutip.

    .. [Chow] Chow et al., 2012, https://doi.org/10.1103/PhysRevLett.109.060501

    :param numpy.ndarray pauli_tm: The Pauli Transfer Matrix as 2d-array.
    :param OperatorBasis basis:  The operator basis, typically products of normalized Paulis.
    :return qutip.Qobj: The Choi matrix as qutip.Qobj.
    """

    if not basis.is_orthonormal():  # pragma no coverage
        raise ValueError("Need an orthonormal operator basis.")
    if not all((is_hermitian(op) for op in basis.ops)):  # pragma no coverage
        raise ValueError("Need an operator basis of hermitian operators.")

    sbasis = basis.super_basis()
    D = basis.dim
    choi = sum((pauli_tm[jj, kk] * sbasis.ops[jj + kk * D] for jj in range(D) for kk in range(D)))
    choi.superrep = "choi"
    return choi


def to_realimag(Z):
    """
    Convert a complex hermitian matrix to a real valued doubled up representation, i.e., for
    ``Z = Z_r + 1j * Z_i`` return ``R(Z)``::

        R(Z) = [ Z_r   Z_i]
               [-Z_i   Z_r]

    A complex hermitian matrix ``Z`` with elementwise real and imaginary parts
    ``Z = Z_r + 1j * Z_i`` can be
    isomorphically represented in doubled up form as::

        R(Z) = [ Z_r   Z_i]
               [-Z_i   Z_r]

        R(X)*R(Y) = [ (X_r*Y_r-X_i*Y_i)    (X_r*Y_i + X_i*Y_r)]
                    [-(X_r*Y_i + X_i*Y_r)  (X_r*Y_r-X_i*Y_i)  ]

                  = R(X*Y).

    In particular, ``Z`` is complex positive (semi-)definite iff ``R(Z)`` is real positive
    (semi-)definite.

    :param (qutip.Qobj|scipy.sparse.base.spmatrix) Z:  The operator representation matrix.
    :returns: R(Z) the doubled up representation.
    :rtype: scipy.sparse.csr_matrix
    """
    if isinstance(Z, qt.Qobj):
        Z = Z.data
    if not is_hermitian(Z):
        raise ValueError("Need a hermitian matrix Z")
    return spvstack([sphstack([Z.real, Z.imag]), sphstack([Z.imag.T, Z.real])]).tocsr()

# using the Z-basis for POVM terms allows to easily identify multi-body contributions
# Z_i Z_j ... Z_k to the readout signal. This should lead to sparsity vs the readout signal index
POVM_Z_BASIS = OperatorBasis([("I", I), ("Z", Z)])

# using the Pi-basis for POVM terms allows to easily associate the preparations with individual
# multi-qubit projectors
POVM_PI_BASIS = OperatorBasis([("0", GS), ("1", ES)])
