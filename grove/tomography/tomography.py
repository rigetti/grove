"""
Quantum State and Process Tomography
====================================

TODO: add content here
"""
from __future__ import print_function
import itertools
import logging
import time
from collections import namedtuple, OrderedDict
from itertools import product as cartesian_product

import cvxpy
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import tqdm
from pyquil.gates import I, RX, RY
from pyquil.quil import Program
from scipy.sparse import (vstack as spvstack, csr_matrix, coo_matrix, kron as spkron)

import grove.tomography.utils as ut
from grove.tomography.utils import qI, qX, qY

MAX_QUBITS_STATE_TOMO = 4
MAX_QUBITS_PROCESS_TOMO = MAX_QUBITS_STATE_TOMO // 2

def default_rotations_1q(q):
    """
    Generate the QUIL programs for tomographic pre- and post-rotations
    of a single qubit.
    """
    for g in TOMOGRAPHY_GATES:
        p = Program(g(q))
        yield p


def default_rotations(*qubits):
    """
    Generate the QUIL programs for the tomographic pre- and post-rotations
    of any number of qubits.
    """
    for programs in cartesian_product(TOMOGRAPHY_GATES, repeat=len(qubits)):
        p = Program()
        for q, g in zip(qubits, programs):
            p.inst(g(q))
        yield p


def default_channel_ops(nqubits):
    """
    Generate the tomographic pre- and post-rotations
    of any number of qubits as qutip operators.
    """
    for gates in cartesian_product(TOMOGRAPHY_GATES.values(), repeat=nqubits):
        yield qt.tensor(*gates)


def state_tomography_programs(state_prep, qubits=None, rotation_generator=default_rotations):
    """
    Yield tomographic sequences that prepare a state with QUIL program `state_prep`
    and then append tomographic rotations on the specified `qubits`.

    If `qubits is None`, it assumes all qubits in the program should be
    tomographically rotated.
    """
    if qubits is None:
        qubits = state_prep.get_qubits()
    for tp in rotation_generator(*qubits):
        p = Program()
        p.inst(state_prep)
        p.inst(tp)
        yield p


def process_tomography_programs(proc, qubits=None,
                                pre_rotation_generator=default_rotations,
                                post_rotation_generator=default_rotations):
    """
    Generator that yields tomographic sequences that wrap a process encoded by a QUIL program `proc`
    in tomographic rotations on the specified `qubits`.

    If `qubits is None`, it assumes all qubits in the program should be
    tomographically rotated.

    :param Program proc: A Quil program
    :param list|NoneType qubits: The specific qubits for which to generate the tomographic sequences
    :param pre_rotation_generator:
    :param post_rotation_generator:
    """
    if qubits is None:
        qubits = proc.get_qubits()
    for tp_pre in pre_rotation_generator(*qubits):
        for tp_post in post_rotation_generator(*qubits):
            p = Program()
            p.inst(tp_pre)
            p.inst(proc)
            p.inst(tp_post)
            yield p


_log = logging.getLogger(__name__)


class _SDP_SOLVER(object):
    """
    Helper object that allows to test whether a working convex solver with SDP capabilities is
    installed. Not all solvers supported by cvxpy support positivity constraints.

    Usage:

        if _SDP_SOLVER.is_functional():
            # solve SDP

    """

    _functional = False
    _tested = False

    @classmethod
    def is_functional(cls):
        """
        Checks lazily whether a convex solver is installed that handles positivity constraints.

        :return: True if a solver supporting positivity constraints is installed.
        :rtype: bool
        """
        if not cls._tested:
            cls._tested = True
            np.random.seed(0)
            mat = np.random.randn(10, 10)
            posmat = mat.dot(mat.T)
            posvar = cvxpy.Variable(10, 10)
            prob = cvxpy.Problem(
                cvxpy.Minimize(
                    cvxpy.trace(posmat*posvar) + cvxpy.norm(posvar)
                ),
                [
                     posvar >> 0,
                     cvxpy.trace(posvar) >= 1.,
                ])
            try:
                prob.solve()
                cls._functional = True
            except cvxpy.SolverError:  # pragma no coverage
                _log.warn("No convex SDP solver found. You will not be able to solve"
                          " tomography problems with matrix positivity constraints.")

        return cls._functional


_TomographySettings = namedtuple('TomographySettings',
                                 ['constraints', 'solver_kwargs'])


class TomographySettings(_TomographySettings):
    """
    Encapsulate the TomographySettings, i.e., the constraints to be applied to the Maximum
    Likelihood Estimation and the keyword arguments to be passed to the convex solver.

    In particular, the settings are:

    :param set constraints: The constraints to be applied:
    For state tomography the maximal constraints are `{'positive', 'unit_trace'}`.
    For process tomography the maximal constraints are `{'cpositive', 'trace_preserving'}`.
    :param dict solver_kwargs: Keyword arguments to be passed to the convex solver.
    """
    pass

DEFAULT_SOLVER_KWARGS = dict(verbose=False, max_iters=20000)

DEFAULT_STATE_TOMO_SETTINGS = TomographySettings(
    constraints={'unit_trace'},
    solver_kwargs=DEFAULT_SOLVER_KWARGS
)

DEFAULT_PROCESS_TOMO_SETTINGS = TomographySettings(
    constraints=set('trace_preserving'),
    solver_kwargs=DEFAULT_SOLVER_KWARGS
)


class TomographyBaseError(Exception):
    """
    Base class for errors raised during Tomography analysis.
    """
    pass


class IncompleteTomographyError(TomographyBaseError):
    """
    Raised when a tomography SignalTensor has circuit results that are all 0. indicating that the
    measurement did not complete successfully.
    """
    pass


class BadReadoutPOVM(TomographyBaseError):
    """
    Raised when the tomography analysis fails due to a bad readout calibration.
    """
    pass


def _prepare_C_jk_m(readout_povm, pauli_basis, channel_ops):
    """
    Prepare the coefficient matrix for state tomography. This function uses sparse matrices
    for much greater efficiency.
    The coefficient matrix is defined as:

    .. math::

            C_{(jk)m} = \tr{\Pi_{s_j} \Lambda_k(P_m)} = \sum_{r}\pi_{jr}(\mathcal{R}_{k})_{rm}

    where :math:`\Lambda_k(\cdot)` is the quantum map corresponding to the k-th pre-measurement
    channel, i.e., :math:`\Lambda_k(\rho) = E_k \rho E_k^\dagger` where :math:`E_k` is the k-th
    channel operator. This map can also be represented via its transfer
    matrix :math:`\mathcal{R}_{k}`. In that case one also requires the overlap
    between the (generalized) Pauli basis ops and the projection operators
    :math:`\pi_{jl}:=\sbraket{\Pi_j}{P_l} = \tr{\Pi_j P_l}`.


    See the willow documentation on tomography for detailed information.

    :param DiagonalPOVM readout_povm: The POVM corresponding to the readout plus classifier.
    :param OperatorBasis pauli_basis: The (generalized) Pauli basis employed in the estimation.
    :param list channel_ops: The pre-measurement channel operators as `qutip.Qobj`
    :return: The coefficient matrix necessary to set up the binomial state tomography problem.
    :rtype: scipy.sparse.csr_matrix
    """

    channel_transfer_matrices = [pauli_basis.transfer_matrix(qt.to_super(Ek)) for Ek in channel_ops]

    # this bit could be more efficient but does not run super long and is thus
    # preserved for readability.
    pi_jr = csr_matrix(
        [pauli_basis.project_op(N_j).toarray().ravel()
         for N_j in readout_povm.ops])

    # Dict used for constructing our sparse matrix, keys are tuples (row_index, col_index), values
    # are the non-zero elements of the final matrix.
    C_jk_m_elms = {}

    # This explicitly exploits the sparsity of all operators involved
    for k in range(len(channel_ops)):
        pi_jr_Rk_rm = (pi_jr * channel_transfer_matrices[k]).tocoo()
        for (j, m, val) in itertools.izip(pi_jr_Rk_rm.row, pi_jr_Rk_rm.col, pi_jr_Rk_rm.data):
            # The multi-index (j,k) is enumerated in column-major ordering (like Fortran arrays)
            C_jk_m_elms[(j + k * readout_povm.pi_basis.dim, m)] = val

    # create sparse matrix from COO-format (see scipy.sparse docs)
    _keys, _values = itertools.izip(*C_jk_m_elms.items())
    _rows, _cols = itertools.izip(*_keys)
    C_jk_m = coo_matrix((list(_values), (list(_rows), list(_cols))),
                        shape=(readout_povm.pi_basis.dim * len(channel_ops), pauli_basis.dim)).tocsr()
    return C_jk_m


def _prepare_B_jkl_mn(readout_povm, pauli_basis, pre_channel_ops, post_channel_ops, rho0):
    """
    Prepare the coefficient matrix for process tomography. This function uses sparse matrices
    for much greater efficiency.
    The coefficient matrix is defined as:

    .. math::

            B_{(jkl)(mn)}=\sum_{r,q}\pi_{jr}(\mathcal{R}_{k})_{rm} (\mathcal{R}_{l})_{nq} (\rho_0)_q

    where :math:`\mathcal{R}_{k}` is the transfer matrix of the quantum map corresponding to the
    k-th pre-measurement channel, while :math:`\mathcal{R}_{l}` is the transfer matrix of the l-th
    state preparation process. We also require the overlap
    between the (generalized) Pauli basis ops and the projection operators
    :math:`\pi_{jl}:=\sbraket{\Pi_j}{P_l} = \tr{\Pi_j P_l}`.

    See the willow documentation on tomography for detailed information.

    :param DiagonalPOVM readout_povm: The POVM corresponding to the readout plus classifier.
    :param OperatorBasis pauli_basis: The (generalized) Pauli basis employed in the estimation.
    :param list pre_channel_ops: The state preparation channel operators as `qutip.Qobj`
    :param list post_channel_ops: The pre-measurement (post circuit) channel operators as `qutip.Qobj`
    :param qutip.Qobj rho0: The initial state as a density matrix.
    :return: The coefficient matrix necessary to set up the binomial state tomography problem.
    :rtype: scipy.sparse.csr_matrix
    """
    C_jk_m = _prepare_C_jk_m(readout_povm, pauli_basis, post_channel_ops)

    pre_channel_transfer_matrices = [pauli_basis.transfer_matrix(qt.to_super(Ek))
                                    for Ek in pre_channel_ops]
    rho0_q = pauli_basis.project_op(rho0)

    # These next lines hide some very serious (sparse-)matrix index magic,
    # basically we exploit the same index math as in `qutip.sprepost()`
    # i.e., if a matrix X is linearly mapped `X -> A.dot(X).dot(B)`
    # then this can be rewritten as
    #           `np.kron(B.T, A).dot(X.T.ravel()).reshape((B.shape[1], A.shape[0])).T`
    # The extra matrix transpose operations are necessary because numpy by default
    # uses row-major storage, whereas these operations are conventionally defined for column-major
    # storage.
    D_ln = spvstack([(Rlnq * rho0_q).T for Rlnq in pre_channel_transfer_matrices]).tocoo()
    B_jkl_mn = spkron(D_ln, C_jk_m).real

    return B_jkl_mn


class StateTomography(object):
    """
    A StateTomography object encapsulates the result of quantum state estimation from tomographic
    data. It provides convenience functions for visualization and computing state fidelities.
    """

    @staticmethod
    def estimate_from_ssr(histograms, readout_povm, channel_ops, settings):
        """
        Estimate a density matrix from single shot histograms obtained by measuring bitstrings in
        the Z-eigenbasis after application of given channel operators.

        :param numpy.ndarray histograms: The single shot histograms, `shape=(n_channels, dim)`.
        :param DiagognalPOVM readout_povm: The POVM corresponding to the readout plus classifier.
        :param list channel_ops: The tomography measurement channels as `qutip.Qobj`'s.
        :param TomographySettings settings: The solver and estimation settings.
        :return: The generated StateTomography object.
        :rtype: StateTomography
        """
        nqc = len(channel_ops[0].dims[0])

        pauli_basis = ut.PAULI_BASIS ** nqc
        pi_basis = readout_povm.pi_basis

        if not histograms.shape[1] == pi_basis.dim:  # pragma no coverage
            raise ValueError("Currently tomography is only implemented for two-level systems")

        # prepare the log-likelihood function parameters, see documentation
        n_kj = np.asarray(histograms)
        C_jk_m = _prepare_C_jk_m(readout_povm, pauli_basis, channel_ops)
        rho_m = cvxpy.Variable(pauli_basis.dim)

        p_jk = C_jk_m * rho_m
        obj = -1.0 * n_kj.ravel() * cvxpy.log(p_jk)

        p_jk_mat = cvxpy.reshape(p_jk, pi_basis.dim, len(channel_ops))   # cvxpy has col-major order

        # Default constraints:
        # MLE must describe valid probability distribution
        # i.e., for each k, p_jk must sum to one and be element-wise non-negative:
        # 1. \sum_j p_jk == 1  for all k
        # 2. p_jk >= 0         for all j, k
        # where p_jk = \sum_m C_jk_m rho_m
        constraints = [
            p_jk >= 0,
            np.matrix(np.ones((1, pi_basis.dim))) * p_jk_mat == 1,
        ]

        rho_m_real_imag = sum((rm * ut.to_realimag(Pm)
                               for (rm, Pm) in zip(rho_m, pauli_basis.ops)), 0)

        if 'positive' in settings.constraints:
            if _SDP_SOLVER.is_functional():
                constraints.append(rho_m_real_imag >> 0)
            else:  # pragma no coverage
                _log.warn("No convex solver capable of semi-definite problems installed.\n"
                          "Dropping the positivity constraint on the density matrix")

        if 'unit_trace' in settings.constraints:
            # this assumes that the first element of the Pauli basis is always proportional to
            # the identity
            constraints.append(rho_m[0,0] == 1. / pauli_basis.ops[0].tr().real)

        prob = cvxpy.Problem(cvxpy.Minimize(obj), constraints)

        _log.info("Starting convex solver")
        prob.solve(**settings.solver_kwargs)
        if prob.status != cvxpy.OPTIMAL:  # pragma no coverage
            _log.warn("Problem did not converge to optimal solution. "
                      "Solver settings: {}".format(settings.solver_kwargs))

        return StateTomography(np.array(rho_m.value).ravel(), pauli_basis, settings)

    def __init__(self, rho_coeffs, pauli_basis, settings):
        """
        Construct a StateTomography to encapsulate the result of estimating the quantum state from
        a quantum tomography measurement.

        :param numpy.ndarray R_est: The estimated quantum state represented in a given (generalized)]
        Pauli basis.
        :param OperatorBasis pauli_basis: The employed (generalized) Pauli basis.
        :param TomographySettings settings: The settings used to estimate the state.
        """
        self.rho_coeffs = rho_coeffs
        self.pauli_basis = pauli_basis
        self.rho_est = sum((r_m * P_m for r_m, P_m in zip(rho_coeffs, pauli_basis.ops)))
        self.settings = settings

    def fidelity(self, other):
        """
        Compute the quantum state fidelity of the estimated state with another state.

        :param qutip.Qobj other: The other quantum state.
        :return: The fidelity, a real number between 0 and 1.
        :rtype: float
        """
        return qt.fidelity(self.rho_est, other)

    def plot_state_histogram(self, ax):
        """
        Visualize the complex matrix elements of the estimated state.

        :param matplotlib.Axes ax: A matplotlib Axes object to plot into.
        """
        title = "Estimated state"
        if hasattr(self, 'fidelity_vs_qvm'):
            title += r', $Fidelity={:1.2g}$'.format(self.fidelity_vs_qvm)
        nqc = int(round(np.log2(self.rho_est.data.shape[0])))
        labels = ut.basis_labels(nqc)
        return ut.state_histogram(self.rho_est,
                                  ax,
                                  title,
                                  xlabels=labels,
                                  ylabels=labels)

    def plot(self):
        """
        Visualize the state.

        :return: The generated figure.
        :rtype: matplotlib.Figure
        """
        f = plt.figure(figsize=(10, 8))
        ax = f.add_subplot(111, projection="3d")
        self.plot_state_histogram(ax)
        return f


class ProcessTomography(object):
    """
    A ProcessTomography object encapsulates the result of quantum process estimation from
    tomographic data. It provides convenience functions for visualization, computing process
    fidelities and inter-conversion between different numerical representation of quantum processes.
    """

    @staticmethod
    def estimate_from_ssr(histograms, readout_povm, pre_channel_ops, post_channel_ops, settings):
        """
        Estimate a quantum process from single shot histograms obtained by preparing specific input
        states and measuring bitstrings in the Z-eigenbasis after application of given channel
        operators.

        :param numpy.ndarray histograms: The single shot histograms.
        :param DiagonalPOVM readout_povm: The POVM corresponding to readout plus classifier.
        :param list pre_channel_ops: The input state preparation channels as `qutip.Qobj`'s.
        :param list post_channel_ops: The tomography post-process channels as `qutip.Qobj`'s.
        :param TomographySettings settings: The solver and estimation settings.
        """
        nqc = len(pre_channel_ops[0].dims[0])
        pauli_basis = ut.PAULI_BASIS ** nqc
        pi_basis = readout_povm.pi_basis

        if not histograms.shape[-1] == pi_basis.dim:  # pragma no coverage
            raise ValueError("Currently tomography is only implemented for two-level systems")

        rho0 = ut.n_qubit_ground_state(nqc)

        n_lkj = np.asarray(histograms)

        B_jkl_mn = _prepare_B_jkl_mn(readout_povm, pauli_basis, pre_channel_ops,
                                     post_channel_ops, rho0)

        R_mn = cvxpy.Variable(pauli_basis.dim ** 2)
        p_jkl = B_jkl_mn.real * R_mn
        obj = -1.0 * np.matrix(n_lkj.ravel()) * cvxpy.log(p_jkl)

        # cvxpy has col-major order and we collapse k and l onto single dimension
        p_jkl_mat = cvxpy.reshape(p_jkl, pi_basis.dim, len(pre_channel_ops) * len(post_channel_ops))

        # Default constraints:
        # MLE must describe valid probability distribution
        # i.e., for each k and l, p_jkl must sum to one and be element-wise non-negative:
        # 1. \sum_j p_jkl == 1  for all k, l
        # 2. p_jkl >= 0         for all j, k, l
        # where p_jkl = \sum_m B_jkl_mn R_mn
        constraints = [
            p_jkl >= 0,
            np.matrix(np.ones((1, pi_basis.dim))) * p_jkl_mat == 1,
        ]

        R_mn_mat = cvxpy.reshape(R_mn, pauli_basis.dim, pauli_basis.dim)

        super_pauli_basis = pauli_basis.super_basis()

        choi_real_imag = sum((R_mn_mat[jj, kk] * ut.to_realimag(
            super_pauli_basis.ops[jj + kk * pauli_basis.dim])
                                   for jj in range(pauli_basis.dim)
                                   for kk in range(pauli_basis.dim)), 0)

        if 'cpositive' in settings.constraints:
            if _SDP_SOLVER.is_functional():
                constraints.append(choi_real_imag >> 0)
            else:  # pragma no coverage
                _log.warn("No convex solver capable of semi-definite problems installed.\n"
                          "Dropping the complete positivity constraint on the process")

        if 'trace_preserving' in settings.constraints:
            constraints.append(R_mn_mat[0, 0] == 1)
            constraints.append(R_mn_mat[0, 1:] == 0)

        prob = cvxpy.Problem(cvxpy.Minimize(obj), constraints)
        sol = prob.solve(**settings.solver_kwargs)

        R_mn_est = R_mn.value.reshape((pauli_basis.dim, pauli_basis.dim)).transpose()
        return ProcessTomography(R_mn_est, pauli_basis, settings)

    def __init__(self, R_est, pauli_basis, settings):
        """
        Construct a ProcessTomography to encapsulate the result of estimating a quantum process
        from a quantum tomography measurement.

        :param numpy.ndarray R_est: The estimated quantum process represented as a Pauli transfer
        matrix.
        :param OperatorBasis pauli_basis: The employed (generalized) Pauli basis.
        :param TomographySettings settings: The settings used to estimate the process.
        """
        self.R_est = R_est
        self.sop = pauli_basis.super_from_tm(R_est)
        self.pauli_basis = pauli_basis
        self.settings = settings

    def process_fidelity(self, other):
        """
        Compute the quantum process fidelity of the estimated state with respect to a unitary
        process.

        :param (qutip.Qobj|matrix-like) other: A unitary operator that induces a process as
            ``rho -> other*rho*other.dag()``, can also be a superoperator or Pauli-transfer matrix.
        :return: The process fidelity, a real number between 0 and 1.
        :rtype: float
        """
        if isinstance(other, qt.Qobj):
            if not other.issuper or other.superrep != "super":
                sother = qt.to_super(other)
            else:
                sother = other
            tm_other = self.pauli_basis.transfer_matrix(sother)
        else:
            tm_other = csr_matrix(other)
        d = self.pauli_basis.ops[0].shape[0]
        return np.trace(tm_other.T*self.R_est).real / d**2

    def avg_gate_fidelity(self, other):
        """
        Compute the average gate fidelity of the estimated state with respect to a unitary process.
        See `Chow et al., 2012, <https://doi.org/10.1103/PhysRevLett.109.060501>`_

        :param (qutip.Qobj|matrix-like) other: A unitary operator that induces a process as
            `rho -> other*rho*other.dag()`, alternatively a superoperator or Pauli-transfer matrix.
        :return: The average gate fidelity, a real number between 1/(d+1) and 1, where d is the
        Hilbert space dimension.
        :rtype: float
        """
        pf = self.process_fidelity(other)
        d = self.pauli_basis.ops[0].shape[0]
        return (d*pf + 1.)/(d + 1.)

    def to_super(self):
        """
        Compute the standard superoperator representation of the estimated process.

        :return: The process as a superoperator.
        :rytpe: qutip.Qobj
        """
        return self.sop

    def to_choi(self):
        """
        Compute the choi matrix representation of the estimated process.

        :return: The process as a choi-matrix.
        :rytpe: qutip.Qobj
        """
        return qt.to_choi(self.sop)

    def to_chi(self):
        """
        Compute the chi process matrix representation of the estimated process.

        :return: The process as a chi-matrix.
        :rytpe: qutip.Qobj
        """
        return qt.to_chi(self.sop)

    def plot_pauli_transfer_matrix(self, ax):
        """
        Plot the elements of the Pauli transfer matrix.

        :param matplotlib.Axes ax: A matplotlib Axes object to plot into.
        """
        title = "Estimated process"
        if hasattr(self, 'fidelity_vs_qvm'):
            title += r', $F_{{\rm avg}}={:1.2g}$'.format(self.fidelity_vs_qvm)
        ut.plot_pauli_transfer_matrix(self.R_est,
                                      ax,
                                      self.pauli_basis.labels,
                                      title)

    def plot(self):
        """
        Visualize the process.

        :return: The generated figure.
        :rtype: matplotlib.Figure
        """
        f, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
        self.plot_pauli_transfer_matrix(ax1)
        return f


TOMOGRAPHY_GATES = OrderedDict([
    (I, qI),
    (RX(np.pi/2), (-1j * np.pi / 4 * qX).expm()),
    (RY(np.pi/2), (-1j * np.pi / 4 * qY).expm()),
    (RX(np.pi), (-1j * np.pi / 2 * qX).expm())
])


def bitstring_to_int(bitstring):
    """Convert a binary bitstring into the corresponding unsigned integer.
    """
    ret = 0
    for b in bitstring:
        ret = (ret << 1) | (int(b) & 1)
    return ret


def wait_until_done(result):
    if isinstance(result, list):
        return result
    while True:
        if result.is_done():
            return result.result['result']
        time.sleep(.1)


def sample_assignment_probs(qubits, nsamples, cxn):
    nq = len(qubits)
    d = 2 ** nq
    hists = []
    preps = ut.basis_state_preps(*qubits)
    for jj, p in zip(tqdm.trange(d), preps):
        results = cxn.run_and_measure(p, qubits, nsamples)
        idxs = map(bitstring_to_int, results)
        hists.append(ut.make_histogram(idxs, d))
    return ut.estimate_assignment_probs(hists)


def do_state_tomography(preparation_program, nsamples, cxn, qubits=None):
    """
    Method to perform both a QPU and QVM state tomography, and use the latter as
    as reference to calculate the fidelity of the former.

    :param Program preparation_program: Program to execute.
    :param int nsamples: Number of samples to take for the program.
    :param QVMConnection|QPUConnection cxn: Connection on which to run the program.
    :param list qubits: List of qubits for the program.
    to use in the tomography analysis.
    :return: The state tomogram
    :rtype: StateTomography
    """

    if qubits is None:
        qubits = sorted(preparation_program.get_qubits())

    if len(qubits) > MAX_QUBITS_STATE_TOMO:
        raise ValueError("Too many qubits!")

    nq = len(qubits)
    d = 2 ** nq

    assignment_probs = sample_assignment_probs(qubits, nsamples, cxn)

    tomo_seq = list(state_tomography_programs(preparation_program, qubits))
    histograms = np.zeros((len(tomo_seq), d))
    for jj, p in zip(tqdm.trange(len(tomo_seq)), tomo_seq):
        results = cxn.run_and_measure(p, qubits, nsamples)
        idxs = map(bitstring_to_int, results)
        histograms[jj] = ut.make_histogram(idxs, d)


    povm = ut.make_diagonal_povm(ut.POVM_PI_BASIS ** nq, assignment_probs)
    channel_ops = list(default_channel_ops(nq))

    state_tomo = StateTomography.estimate_from_ssr(histograms, povm,
                                                   channel_ops,
                                                   DEFAULT_STATE_TOMO_SETTINGS)

    return state_tomo, assignment_probs, histograms


def do_state_tomography_with_qvm_reference(preparation_program,
                                           nsamples,
                                           qpu=None,
                                           qvm=None,
                                           qubits=None):
    """
    Method to perform both a QPU and QVM state tomography, and use the latter as
    as reference to calculate the fidelity of the former.

    :param Program preparation_program: Program to execute.
    :param int nsamples: Number of samples to take for the program.
    :param QPUConnection qpu: QPUConnection on which to run the program.
    :param QVMConnection qvm: QVMConnection on which to run the reference program.
    :param list qubits: List of qubits for the program.
    to use in the tomography analysis.
    :return: The state tomogram
    :rtype: StateTomography
    """
    if qpu is None:
        print("Please provide a QPU connection.")
        return None

    print("Running state tomography on the QPU...")
    tomography_qpu, _, _ = do_state_tomography(preparation_program, nsamples, qpu, qubits)
    print("State tomography completed.")

    if qvm is None:
        print("Warning: Cannot report state fidelity without a QVM on which to \
              calculate the ideal state.")
        return tomography_qpu

    print("Running state tomography on the QVM for reference...")
    tomography_qvm, _, _ = do_state_tomography(preparation_program, nsamples, qvm, qubits)
    print("State tomography completed.")

    # Used for supplying a reference unitary for calculating the fidelity, in plotting
    tomography_qpu.fidelity_vs_qvm = tomography_qpu.fidelity(tomography_qvm.rho_est)
    return tomography_qpu


def do_process_tomography(process, nsamples, cxn, qubits=None):
    """
    Method to perform a process tomography.

    :param Program process: Process to execute.
    :param int nsamples: Number of samples to take for the program.
    :param QVMConnection|QPUConnection cxn: Connection on which to run the program.
    :param list qubits: List of qubits for the program.
    to use in the tomography analysis.
    :return: The process tomogram
    :rtype: ProcessTomography
    """
    if qubits is None:
        qubits = sorted(process.get_qubits())

    nq = len(qubits)
    d = 2 ** nq

    if nq > MAX_QUBITS_PROCESS_TOMO:
        raise ValueError("Too many qubits!")

    assignment_probs = sample_assignment_probs(qubits, nsamples, cxn)

    tomo_seq = list(process_tomography_programs(process, qubits))
    histograms = np.zeros((len(tomo_seq), d))
    for jj, p in zip(tqdm.trange(len(tomo_seq)), tomo_seq):
        results = wait_until_done(cxn.run_and_measure(p, qubits, nsamples))
        idxs = map(bitstring_to_int, results)
        histograms[jj] = ut.make_histogram(idxs, d)

    povm = ut.make_diagonal_povm(ut.POVM_PI_BASIS ** nq, assignment_probs)
    channel_ops = list(default_channel_ops(nq))
    histograms = histograms.reshape((len(channel_ops), len(channel_ops), d))
    process_tomo = ProcessTomography.estimate_from_ssr(histograms, povm,
                                                       channel_ops, channel_ops,
                                                       DEFAULT_PROCESS_TOMO_SETTINGS)

    return process_tomo, assignment_probs, histograms


def do_process_tomography_with_qvm_reference(process,
                                             nsamples,
                                             qpu=None,
                                             qvm=None,
                                             qubits=None):
    """
    Method to perform both a QPU and QVM process tomography, and use the latter as
    as reference to calculate the fidelity of the former.

    :param Program process: Process to execute.
    :param int nsamples: Number of samples to take for the program.
    :param QPUConnection qpu: QPUConnection on which to run the program.
    :param QVMConnection qvm: QVMConnection on which to run the reference program.
    :param list qubits: List of qubits for the program.
    to use in the tomography analysis.
    :return: The process tomogram
    :rtype: ProcessTomography
    """
    if qpu is None:
        print("Please provide a QPU connection.")
        return None

    print("Running process tomography on the QPU...")
    tomography_qpu, _, _ = do_process_tomography(process, nsamples, qpu, qubits)
    print("Process tomography completed.")

    if qvm is None:
        print("Warning: Cannot report process fidelity without a QVM on which to \
              calculate the ideal process.")
        return tomography_qpu

    print("Running process tomography on the QVM for reference...")
    tomography_qvm, _, _ = do_process_tomography(process, nsamples, qvm, qubits)
    print("Process tomography completed.")

    # Used for supplying a reference unitary for calculating the fidelity, in plotting
    tomography_qpu.fidelity_vs_qvm = tomography_qpu.avg_gate_fidelity(tomography_qvm.R_est)
    return tomography_qpu
