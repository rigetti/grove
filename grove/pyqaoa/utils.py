##############################################################################
# Copyright 2016-2017 Rigetti Computing
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

from pyquil.quil import Program


def isclose(a, b, rel_tol=1e-10, abs_tol=0.0):
    """
    Compares two parameter values.
    :param a: First parameter
    :param b: Second parameter
    :param rel_tol: Relative tolerance
    :param abs_tol: Absolute tolerance
    :return: Boolean telling whether or not the parameters are close enough to be the same
    """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def compare_progs(test, reference):
    """
    Compares two programs gate by gate, param by param.
    :param Program test: Test program
    :param Program reference: Reference program
    """
    tinstr = test.actions
    rinstr = reference.actions
    assert len(tinstr) == len(rinstr)
    for idx in xrange(len(tinstr)):
        # check each field of the instruction object
        assert tinstr[idx][1].operator_name == rinstr[idx][1].operator_name
        assert len(tinstr[idx][1].parameters) == len(rinstr[idx][1].parameters)
        for pp in xrange(len(tinstr[idx][1].parameters)):
            cmp_val = isclose(tinstr[idx][1].parameters[pp], rinstr[idx][1].parameters[pp])
            assert cmp_val

        assert len(tinstr[idx][1].arguments) == len(rinstr[idx][1].arguments)
        for aa in xrange(len(tinstr[idx][1].arguments)):
            assert tinstr[idx][1].arguments[aa] == rinstr[idx][1].arguments[aa]
