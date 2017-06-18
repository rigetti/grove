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

import pytest
import collections
from grove.order_finding.shor import factor
compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

def test_shor_n_zero():
    '''
    Tests that an exception is raised when N = 0 is asked for
    '''
    
    with pytest.raises(ValueError) as excinfo:
        def f():
            factor(0)
        f()
    assert "N = 0 has no prime factorization" in str(excinfo.value)
    
def test_shor_n_negative():
    '''
    Tests that an exception is raised when N < 0 is asked for
    '''
    
    with pytest.raises(ValueError) as excinfo:
        def f():
            factor(-5)
        f()
    assert "The prime factorization of negative numbers is not defined" in str(excinfo.value)

def test_shor_n_not_int():
    '''
    Tests that an exception is raised when N is not an integer
    '''
    
    with pytest.raises(ValueError) as excinfo:
        def f():
            factor(3.14)
        f()
    assert "N must be an integer " in str(excinfo.value)

def test_shor_n_trivial():
    '''
    Tests the factorization of a trivial N (1, 2, or 3)
    '''
    
    n = 2
    result = factor(n)
    assert result == [2]
    
def test_shor_n_prime():
    '''
    Tests the factorization of a prime number
    '''
    
    n = 7
    result = factor(n)
    assert result == [7]
    
def test_shor_n_semiprime_same():
    '''
    Tests the prime factorization of N = p*p, where p is a prime
    '''
    
    n = 7*7
    result = factor(n)
    assert compare(result, [7,7])
    
def test_shor_n_semiprime_distinct():
    '''
    Tests the prime factorization of N = p*q, where p and q are primes
    '''
    
    n = 13*23
    result = factor(n)
    assert compare(result, [13,23])
    
def test_shor_n_composite_distinct():
    '''
    Tests the prime factorization of N=p*q*r, where p, q, and r are primes
    '''
    
    n = 13*23*7
    result = factor(n)
    assert compare(result, [13,23,7])
    
def test_shor_n_composite_same():
    '''
    Tests the prime factorization of N=p*p*p, where p is a prime
    '''
    
    n = 7*7*7
    result = factor(n)
    assert compare(result, [7,7,7])