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
import pytest

from grove.tomography.tomography import _SDP_SOLVER
from grove.tomography.utils import import_cvxpy, import_qutip, notebook_mode, NOTEBOOK_MODE


qt = import_qutip()
cvxpy = import_cvxpy()

if not qt:
    pytest.skip("Qutip not installed, skipping tests", allow_module_level=True)

if not cvxpy:
    pytest.skip("CVXPY not installed, skipping tests", allow_module_level=True)


def test_SDP_SOLVER():
    assert _SDP_SOLVER.is_functional()


def test_notebook_mode():
    for mode in [True, False]:
        notebook_mode(mode)
        assert NOTEBOOK_MODE == mode
