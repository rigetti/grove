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
