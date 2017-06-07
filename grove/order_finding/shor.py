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
import pyquil.forest as forest
from order_finding import calculate_order

def factor(N):
    '''
    Computes the prime factors of N using a modified Shor's algorithm, which
    handles cases where there may exist more than two prime factors. Algorithm 
    and notes can be found at https://en.wikipedia.org/wiki/Shor's_algorithm.
    :param N: The integer N = p*q*r*... to find the prime factors of.
    :return: An array of prime factors of N (including duplicates)
    '''
    
    # Initial check for bad or obvious N
    if N == 0:
        raise ValueError("N = 0 has not prime factorization")
    elif N == 1 or N == 2 or N == 3:
        return [N]
    
    # Create a loop which goes through a series of numbers until all are primes
    factors = [N]
    index = 0
    
    # Within this loop, N is the currently selected value for the factor list
    while True:
        print("In loop ", factors)
        
        # Done if at end of list
        if index == len(factors):
            break
            
        # If prime, move to the next factor in the list
        # TODO: An efficient primality test that runs in less than O(n^3)
        if isPrime(factors[index]):
            index += 1
            continue
        
        # Begin Shor's algorithm
        N = factors[index]
        del factors[index]
        
        # Pick X < N and find the GCD
        X = random.randint(1, N) # Double check this range
        
        gcd_val = gcd(X, N)
        if gcd_val != 1:
            # Good guess on X!
            factors += [gcd_val]
            factors += [N / gcd_val]
            continue
            
        # Continue with the quantum part of Shor's if unlucky with X
        print("Starting order calc")
        r = calculate_order(X, N) # returns the program
        print("Got order result")
        
        # Compute p and q
        a = X**(r/2) + 1
        p = gcd(a, N)
        q = N / p
        
        factors += [p]
        factors += [q]
        
    # Now need to remove unnecessary values
    while 1 in factors:
        factors.remove(1)
        
    return factors
        
        
def isPrime(n):
    '''
    An efficient algorithm which determines if the given number is prime
    :param n: The number to check the primality of
    :return: true if n is prime, and false otherwise
    '''
    if n % 2 == 0:
        return False
    i = 3
    while i*i <= n:
        if n % i == 0:
            return False
        i += 2
    return True
    
def gcd(a, b):
    '''
    Finds the greatest common denominator of a and b
    :param a: The first value to consider
    :param b: The second value to consider
    :return: The greatest common denominator of a and b
    '''
    print("In gcd")
    
    while b != 0:
        bTemp = b
        b = a % b
        a = bTemp
    return a

print factor(3*7*5)