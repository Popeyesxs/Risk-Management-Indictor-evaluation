#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Dezheng Xu (zh686437@bu.edu)
Description:First function is that write a definition for the base class BSOption, 
which encapsulates the data required to do Black-Scholes option pricing formula.
The Second function is that create the following 4 methods on the class BSOption 
to calculate d1, d2, nd1, and nd2. Third function is that write method declarations 
on class BSOption for the methods value(self) and delta(self). Forth function is that 
write a definition for the class BSEuroCallOption, which inherits from the base 
class BSOPtion and implements the pricing algorithm for a European-style call option.
Fifth function is that write a definition for the class BSEuroPutOption, which inherits 
from the base class BSOPtion and implements the pricing algorithm for a European-style put option.
Sixth function is that add method declarations on the BSEuroCallOption and BSEuroPutOption 
classes to override the base-class implementation of the delta(self) method. 
"""

from scipy.stats import norm
import math
#function 1 

class BSOption:
    def  __init__(self, s, x, t, sigma, rf) :
        """initialize a new BSOption object of this class
           s (the current stock price in dollars),
           x (the option strike price),
           t (the option maturity time in years),
           sigma (the annualized standard deviation of returns),
           rf (the annualized risk free rate of return),
           div (the annualized dividend rate; assume continuous dividends rate),
        """
        self.s = s
        self.x = x
        self.t = t
        self.sigma = sigma
        self.rf = rf

    def __repr__(self):
        """return a beautifully-formatted string representation of the BSOption object
        """
        s = "s = $%.2f, x = $%.2f, t = %.2f (years), sigma = %.3f, rf = %.3f" %(self.s, self.x, self.t, self.sigma, self.rf)
        return s
    
#function 2


    def d1(self):
        """the class BSOption to calculate d1
        """
        f = (self.rf  + (self.sigma ** (2)) / 2 ) * self.t
        return (1/(self.sigma * (self.t ** (0.5)))) *(math.log(self.s/self.x) + f)
    def d2(self):
        """the class BSOption to calculate d2
        """
        d1 = self.d1()
        return  d1 - self.sigma * (self.t **(0.5))
    def nd1(self):
        """the class BSOption to calculate N(d1)
        """
        d1 = self.d1()
        return norm.cdf(d1)
    def nd2(self):
        """the class BSOption to calculate N(d2)
        """
        d2 = self.d2()
        return norm.cdf(d2)

#function 3
    
    def value(self):
        
        """calculate value for base class BSOption
        """
        
        print("Cannot calculate value for base class BSOption." )
        return 0
    
    def delta(self):
        
        """calculate delta for base class BSOption
        """
        
        print("Cannot calculate delta for base class BSOption." )
        return 0
        
    
#function 4 
class BSEuroCallOption(BSOption):
    def __init__(self, s, x, t, sigma, rf):
        """initialize a new BSEuroCallOption object of this class
        """
        
        
        super().__init__(s, x, t, sigma, rf)
    def value(self):
        """implements the pricing algorithm for a European-style call option.
        """
        nd1 = super().nd1()
        nd2 = super().nd2()
        f1 = nd1 * self.s
        f2 = nd2 * self.x * math.e ** (-self.rf * self.t)
        return f1 - f2
    
    def __repr__(self):
        """return a beautifully-formatted string representation of the BSEuroCallOption object
        """
        s = "BSEuroCallOption, value = $%.2f, \n" %(self.value())
        s += "parameters = (s = $%.2f, x = $%.2f, t = %.2f (years), sigma = %.3f, rf = %.3f) " %(self.s, self.x, self.t, self.sigma, self.rf)
        return s

#function 6_1
    def delta(self):
        """create a hedging portfolio to offset the option’s price risk
        """
        return super().nd1()
    

#function 5 
class BSEuroPutOption(BSOption):
    
    def __init__(self, s, x, t, sigma, rf):
        
        """initialize a new BSEuroPutOption object of this class
        """
        super().__init__(s, x, t, sigma, rf)
    
    def value(self):
        """implements the pricing algorithm for a European-style put option.
        """
        nd1 = super().nd1()
        nd2 = super().nd2()
        _nd1 = 1 - nd1
        _nd2 = 1 - nd2
        f1 = _nd1 * self.s
        f2 = _nd2 * self.x * math.exp(-self.rf * self.t)
        return f2 - f1
        
    def __repr__(self):
        """return a beautifully-formatted string representation of the BSEuroPutOption object
        """
        s = "BSEuroPutOption, value = $%.2f, \n" %(self.value())
        s += "parameters = (s = $%.2f, x = $%.2f, t = %.2f (years), sigma = %.3f, rf = %.3f) " %(self.s, self.x, self.t, self.sigma, self.rf)
        return s

#function 6_1
    def delta(self):
        """create a hedging portfolio to offset the option’s price risk
        """
        return super().nd1() - 1
    
if __name__ == '__main__':
    call = BSEuroCallOption(100, 100, 0.5, 0.25, 0.04)
    call.delta()
    put = BSEuroPutOption(100, 100, 0.5, 0.25, 0.04)
    put.delta()
    call.sigma = 0.5
    call.delta()
    put.sigma = 0.5
    put.delta()

    
    

    
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    