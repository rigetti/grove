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

import random
import primefac
from order_finding import calculate_order, gcd

def factor(N, verbose=False):
    '''
    Computes the prime factors of N using a modified Shor's algorithm, which
    handles cases where there may exist more than two prime factors. Algorithm 
    and notes can be found at https://en.wikipedia.org/wiki/Shor's_algorithm.

    :param N: The integer N = p*q*r*... to find the prime factors of.
    :return: An array of prime factors of N (including duplicates)
    '''
    
    # Initial check for bad or obvious N
    if N == 0:
        raise ValueError("N = 0 has no prime factorization")
    elif N < 0:
        raise ValueError("The prime factorization of negative numbers is not defined (your N = " + str(N) + ")")
    elif int(N) != N:
        raise ValueError("N must be an integer (your N = " + str(N) + ")")
    elif N == 1 or N == 2 or N == 3:
        return [N]
    
    # Create a loop which goes through a series of numbers until all are primes
    factors = [N]
    index = 0
    
    # Within this loop, N is the currently selected value for the factor list
    while True:
        
        # Done if at end of list
        if index == len(factors):
            break
            
        # If prime, move to the next factor in the list
        if primefac.isprime(factors[index]):
            if verbose:
                print "Found prime factor p =", factors[index]
            index += 1
            continue
        
        # Begin Shor's algorithm
        N = factors[index]
        del factors[index]
        
        # Pick X < N and find the GCD
        X = random.randint(1, N) # Double check this range
        if verbose:
            print "Trying modular base X =", X
        
        gcd_val = gcd(X, N)
        if gcd_val != 1:
            # Good guess on X!
            if verbose:
                print "Good guess! GCD produced a factor."
            factors += [gcd_val]
            factors += [N / gcd_val]
            continue
            
        # Continue with the quantum part of Shor's if unlucky with X
        if verbose:
            print "Begininning quantum order finding subroutine."
        r = calculate_order(X, N, verbose=verbose) # returns the program
        if r > N:
            factors.append(N)
            continue
        if verbose:
            print "Calculated order r =", r
        
        # Compute p and q
        a = X**(r/2) + 1
        p = gcd(a, N)
        q = N / p
        
        # If p or q are not integers, restart (most likely indicates noisy r)
        if p != int(p) or q != int(q):
            factors.append(N)
            continue

        factors += [p]
        factors += [q]
        
    # Now need to remove unnecessary values
    while 1 in factors:
        factors.remove(1)
        
    return factors


if __name__ == "__main__":
    N = input("Enter a number to factor: ")
    factors = factor(N, True)
    print "Factors of", N, "are", factors
    assert reduce(int.__mul__, factors) == N, "Try again with more iterations of the order finding subroutine"

