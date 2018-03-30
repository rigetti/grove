##############################################################################
# Copyright 2017-2018 Rigetti Computing
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
##############################################################################

from collections import OrderedDict, namedtuple
import logging
import itertools

from grove.tomography.utils import to_density_matrix, import_qutip

try:
    # Python 2
    from itertools import izip
except ImportError:  # pragma no coverage
    # Python 3
    izip = zip

import numpy as np
from scipy.sparse import hstack as sphstack, vstack as spvstack
from scipy.sparse.linalg import norm as spnorm


_log = logging.getLogger(__name__)

qt = import_qutip()


FROBENIUS = 'fro'
CHOI = 'choi'

EPS = 1e-8


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

    See also the grove documentation.

    :param OperatorBasis pi_basis: An operator basis of rank-1 projection operators.
    :param numpy.ndarray confusion_rate_matrix: The matrix of detection probabilities conditional
    on a prepared qubit state.
    :return: The POVM corresponding to confusion_rate_matrix.
    :rtype: DiagonalPOVM
    """

    confusion_rate_matrix = np.asarray(confusion_rate_matrix)
    if not np.allclose(confusion_rate_matrix.sum(axis=0), np.ones(confusion_rate_matrix.shape[1])):
        raise CRMUnnormalizedError("Unnormalized confusion matrix:\n{}".format(
            confusion_rate_matrix))
    if not (confusion_rate_matrix >= 0).all() or not (confusion_rate_matrix <= 1).all():
        raise CRMValueError("Confusion matrix must have values in [0, 1]:"
                            "\n{}".format(confusion_rate_matrix))

    ops = [sum((pi_j * pjk for (pi_j, pjk) in izip(pi_basis.ops, pjs)), 0)
           for pjs in confusion_rate_matrix]
    return DiagonalPOVM(pi_basis=pi_basis, confusion_rate_matrix=confusion_rate_matrix, ops=ops)


def is_hermitian(operator):
    """
    Check if matrix or operator is hermitian.

    :param (numpy.ndarray|qutip.Qobj) operator: The operator or matrix to be tested.
    :return: True if the operator is hermitian.
    :rtype: bool
    """
    if isinstance(operator, qt.Qobj):
        return (operator.dag() - operator).norm(FROBENIUS) / operator.norm(FROBENIUS) < EPS
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
    return (is_hermitian(operator) and (operator * operator - operator).norm(FROBENIUS)
            / operator.norm(FROBENIUS) < EPS)


def choi_matrix(pauli_tm, basis):
    """
    Compute the Choi matrix for a quantum process from its Pauli Transfer Matrix.

    This agrees with the definition in
    `Chow et al. <https://doi.org/10.1103/PhysRevLett.109.060501>`_
    except for a different overall normalization.
    Our normalization agrees with that of qutip.

    :param numpy.ndarray pauli_tm: The Pauli Transfer Matrix as 2d-array.
    :param OperatorBasis basis:  The operator basis, typically products of normalized Paulis.
    :return: The Choi matrix as qutip.Qobj.
    :rtype: qutip.Qobj
    """

    if not basis.is_orthonormal():  # pragma no coverage
        raise ValueError("Need an orthonormal operator basis.")
    if not all((is_hermitian(op) for op in basis.ops)):  # pragma no coverage
        raise ValueError("Need an operator basis of hermitian operators.")

    sbasis = basis.super_basis()
    D = basis.dim
    choi = sum((pauli_tm[jj, kk] * sbasis.ops[jj + kk * D] for jj in range(D) for kk in range(D)))
    choi.superrep = CHOI
    return choi


def to_realimag(z):
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

    :param (qutip.Qobj|scipy.sparse.base.spmatrix) z:  The operator representation matrix.
    :returns: R(Z) the doubled up representation.
    :rtype: scipy.sparse.csr_matrix
    """
    if isinstance(z, qt.Qobj):
        z = z.data
    if not is_hermitian(z):  # pragma no coverage
        raise ValueError("Need a hermitian matrix z")
    return spvstack([sphstack([z.real, z.imag]), sphstack([z.imag.T, z.real])]).tocsr().real


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
        Compute a matrix of Hilbert-Schmidt inner products for the basis operators, update
        self._metric, and return the value.

        :return: The matrix of inner products.
        :rtype: numpy.matrix
        """
        if self._metric is None:
            _log.debug("Computing and caching operator basis metric")
            self._metric = np.matrix([[(j.dag() * k).tr() for k in self.ops] for j in self.ops])
        return self._metric

    def is_orthonormal(self):
        """
        Compute a matrix of Hilbert-Schmidt inner products for the basis operators, and see if they
        are orthonormal. If they are return True, else, False.

        :return: True if the basis vectors represented by this OperatorBasis are orthonormal, False
         otherwise.
        :rtype: bool
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

        :return: Yields the labels and qutip operators corresponding to the vectors in this basis.
        :rtype: tuple (str, qutip.qobj.Qobj)
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

        :param int n: The number of identical tensor factors.
        :return: The product basis.
        :rtype: OperatorBasis
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

        The follows the definition in
        `Chow et al. <https://doi.org/10.1103/PhysRevLett.109.060501>`_


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

    def transfer_matrix(self, superoperator):
        """
        Compute the transfer matrix :math:`R_{jk} = \tr[P_j sop(P_k)]`.

        :param qutip.Qobj superoperator: The superoperator to transform.
        :return: The transfer matrix in sparse form.
        :rtype: scipy.sparse.csr_matrix
        """
        if not self.is_orthonormal():  # pragma no coverage
            raise ValueError("transfer_matrix() only implemented for orthonormal operator bases.")
        return self.basis_transform.H * superoperator.data * self.basis_transform

    def super_from_tm(self, transfer_matrix):
        """
        Reconstruct a super operator from a transfer matrix representation.
        This inverts `self.transfer_matrix(...)`.

        :param (numpy.ndarray) transfer_matrix: A process in transfer matrix form.
        :return: A qutip super operator.
        :rtype: qutip.Qobj.
        """
        if not self.is_orthonormal():  # pragma no coverage
            raise ValueError("super_from_tm() only implemented for orthonormal operator bases")

        data = self.basis_transform * transfer_matrix * self.basis_transform.H
        sop = qt.Qobj(data, dims=[self.ops[0].dims, self.ops[0].dims])
        sop.superrep = "super"
        return sop

    def __repr__(self):
        return "<span[{}]>".format(",".join(self.labels))

    def __eq__(self, other):
        return (self.labels == other.labels and all(
            [(my_op - o_op).norm(FROBENIUS) < EPS for (my_op, o_op) in zip(self.ops, other.ops)]))


if qt:

    # using the Pi-basis for POVM terms allows to easily associate the preparations with individual
    # multi-qubit projectors

    QX = qt.sigmax()
    QY = qt.sigmay()
    QZ = qt.sigmaz()
    QI = qt.qeye(2)
    GS = to_density_matrix(qt.basis(2, 0))
    ES = to_density_matrix(qt.basis(2, 1))
    POVM_PI_BASIS = OperatorBasis([("0", GS), ("1", ES)])
    PAULI_BASIS = OperatorBasis(
        [("I", QI / np.sqrt(2)), ("X", QX / np.sqrt(2)),
         ("Y", QY / np.sqrt(2)), ("Z", QZ / np.sqrt(2))])

else:  # pragma no coverage
    QX = QY = QZ = QI = GS = ES = None
    POVM_PI_BASIS = PAULI_BASIS = None


def n_qubit_pauli_basis(n):
    """
    Construct the tensor product operator basis of `n` PAULI_BASIS's.

    :param int n: The number of qubits.
    :return: The product Pauli operator basis of `n` qubits
    :rtype: OperatorBasis
    """
    if n >= 1:
        return PAULI_BASIS ** n
    else:  # pragma no coverage
        raise ValueError("n = {} should be at least 1.".format(n))


def n_qubit_ground_state(n):
    """
    Construct the tensor product of `n` ground states `|0>`.

    :param int n: The number of qubits.
    :return: The state `|000...0>` for `n` qubits.
    :rtype: qutip.Qobj
    """
    return qt.tensor(*([GS] * n))
