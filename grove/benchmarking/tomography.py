"""
Quantum State and Process Tomography
====================================

Provides classes for simulated and actual quantum state and process tomography.
Measurements interact with this module through the high level functions `analyze_state_tomography`
and `analyze_process_tomography` which return objects of type `StateTomographyAnalysis` and
`ProcessTomographyAnalysis` respectively. These encapsulate
the estimated state or process and can be used to easily compute fidelities with target states or
processes. These result objects also include the readout calibration analysis as an attribute
`readout` of type `ReadoutAnalysis` which encapsulates information about the readout.

We denote the number of qubits participating in any given measurement by `m`.
The corresponding Hilbert space has dimension `d = 2**m` and density operators have shape `(d, d)`
and thus `D=d**2` matrix elements. We denote the number of amplitudes to be used for tomography
by `nsignals`, the number of shots taken for a run by `num_shots`.

For a measurement model calibration we will typically measure the readout signals
in `nprep >= d` differently prepared states.

For tomography we will always apply `npost` different channel transformations
after the studied tomography circuit.
For process tomography we additionally apply `npre` different channel transformations
before applying the circuit under study.

Additional utilities in this module facilitate handling/converting QuTiP operators
to represent the underlying quantum models.

Tomography Measurement Workflow
-------------------------------

The measurement class `willow.measurements.tomography.Tomography` takes
all the data for state and process tomography experiments on one or more qubits.
It stores unaveraged shot by shot signals for each pulse sequence in a `SignalTensor` object that
also encapsulates the circuits for each pulse sequence.

There are two important estimation problems for a full tomography experiment.

1. The joint readout of all participating qcomplexes has to be calibrated to
   obtain a mapping from the joint qubit state to the observed readout iq-signals.
   To this end a Tomography measurement always requires a preceding ReadoutCalibration measurement,
   that is automatically prepended when running Tomography measurements from the `measure.py`
   script.

2. The actual tomography measurements in which a specific target circuit (ControlSequence) is
   wrapped by additional pre (only process tomography) and post channel
   transforms. These channel transforms are very simple, well characterized pulses,
   typically Clifford gates. The shot by shot readout data of this stage is used
   for the actual tomography by inverting the calibrated mapping obtained through the
   ReadoutCalibration measurement to solve for the underlying quantum state(s) that produced the
   observed readout signals.
   This mapping is not one-to-one between arbitrary states and the observed signals, which is why
   the pre and post channels are needed to rotate the information encoded in the state
   or process into the measurement basis that the readout corresponds to.

"""
import itertools
import logging
from collections import namedtuple

import cvxpy
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.sparse import (vstack as spvstack, csr_matrix, coo_matrix, kron as spkron)

import grove.benchmarking.operator_utils as ut
import pyquil as pq
from itertools import product as cartesian_product


def basis_state_preps(*qubits):
    """
    Generate a sequence of programs that prepares the measurement
    basis states of some set of qubits in the order such that the qubit
    with highest index is iterated over the most quickly:
    E.g., for qubits=(0, 1), it returns the circuits::

        I_0 I_1
        I_0 X_1
        X_0 I_1
        X_0 X_1
    """
    for prep in cartesian_product(["I", "X"], repeat=len(qubits)):
        p = pq.quil.Program()
        for g, q in zip(prep, qubits):
            p.inst(pq.gates.Gate(g, (), [pq.quilbase.unpack_qubit(q)]))
        yield p


def default_rotations_1q(q):
    """
    Generate the QUIL programs for tomographic pre- and post-rotations
    of a single qubit.
    """
    for g in ["I", "X-HALF", "Y-HALF", "X"]:
        p = pq.quil.Program()
        p.inst(pq.quil.Gate(g, (), [pq.quilbase.unpack_qubit(q)]))
        yield p


def default_rotations(*qubits):
    """
    Generate the QUIL programs for the tomographic pre- and post-rotations
    of any number of qubits.
    """
    for programs in cartesian_product(*(default_rotations_1q(q) for q in qubits)):
        p = pq.quil.Program()
        for pp in programs:
            p.inst(pp)
        yield p


def state_tomography_programs(state_prep, qubits=None, rotation_generator=default_rotations):
    """
    Yield tomographic sequences that prepare a state with QUIL program `state_prep`
    and then append tomographic rotations on the specified `qubits`.

    If `qubits is None`, it assumes all qubits in the program should be
    tomographically rotated.
    """
    if qubits is None:
        qubits = state_prep.extract_qubits()
    for tp in rotation_generator(*qubits):
        p = pq.quil.Program()
        p.inst(state_prep)
        p.inst(tp)
        yield p


def process_tomography_programs(proc, qubits=None,
                                pre_rotation_generator=default_rotations,
                                post_rotation_generator=default_rotations
                                ):
    """
    Yield tomographic sequences that wrap a process encoded by a QUIL program `proc`
    in tomographic rotations on the specified `qubits`.

    If `qubits is None`, it assumes all qubits in the program should be
    tomographically rotated.
    """
    if qubits is None:
        qubits = proc.extract_qubits()
    for tp_pre in pre_rotation_generator(*qubits):
        for tp_post in post_rotation_generator(*qubits):
            p = pq.quil.Program()
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
                                 ['constraints', 'classifier_kwargs', 'solver_kwargs'])


class TomographySettings(_TomographySettings):
    """
    Encapsulate the TomographySettings, i.e., the constraints to be applied to the Maximum
    Likelihood Estimation, the keyword arguments to be passed to the classifier,
    and the keyword arguments to be passed to the convex solver.

    In particular, the settings are:

    :param set constraints: The constraints to be applied:
    For state tomography the maximal constraints are `{'positive', 'unit_trace'}`.
    For process tomography the maximal constraints are `{'cpositive', 'trace_preserving'}`.
    :param dict classifer_kwargs: Keyword arguments to be passed to the `classify` method.
    :param dict solver_kwargs: Keyword arguments to be passed to the convex solver.
    """
    pass

DEFAULT_CLASSIFIER_KWARGS = dict(margin=0.)
DEFAULT_SOLVER_KWARGS = dict(verbose=False, max_iters=20000)

DEFAULT_STATE_TOMO_SETTINGS = TomographySettings(
    constraints={'unit_trace'},
    classifier_kwargs=DEFAULT_CLASSIFIER_KWARGS,
    solver_kwargs=DEFAULT_SOLVER_KWARGS
)

DEFAULT_PROCESS_TOMO_SETTINGS = TomographySettings(
    constraints=set(),
    classifier_kwargs=DEFAULT_CLASSIFIER_KWARGS,
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

        # TODO (nik): make compatible with qudit operator bases once the classifier supports this
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

    def plot_pauli_histogram(self, ax):
        """
        Plot the coefficients of the estimated state with respect to an expansion in the
        Pauli-basis.

        :param matplotlib.Axes ax: A matplotlib Axes object to plot into.
        """
        ut.plot_pauli_rep(self.rho_coeffs, ax, self.pauli_basis.labels, "Estimated State")

    def plot_state_histogram(self, ax):
        """
        Visualize the complex matrix elements of the estimated state.

        :param matplotlib.Axes ax: A matplotlib Axes object to plot into.
        """
        return ut.state_histogram(self.rho_est, ax, "Estimated State")

    def plot(self):
        """
        Visualize the state.

        :return: The generated figure.
        :rtype: matplotlib.Figure
        """
        f = plt.figure(figsize=(10, 4))
        ax1 = f.add_subplot(122)
        ax2 = f.add_subplot(121, projection="3d")
        self.plot_pauli_histogram(ax1)
        self.plot_state_histogram(ax2)
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

        # TODO: make compatible with qudit operator bases once the classifier supports this
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
        ut.plot_pauli_transfer_matrix(self.R_est, ax, self.pauli_basis.labels, "Estimated Process")

    def plot(self):
        """
        Visualize the process.

        :return: The generated figure.
        :rtype: matplotlib.Figure
        """
        f, (ax1) = plt.subplots(1, 1, figsize=(10, 4))
        self.plot_pauli_transfer_matrix(ax1)
        return f
