import numpy as np
from numpy import ndarray

from base import Optim

EPS  = 0.3

class GoldenSearch (Optim):
    def __init__ (self, phi : float = 0.68103) -> None : 
        self.phi = phi 

    def _first (self, a, b) -> tuple[float,float] : 
        return b - (b-a)*self.phi, a + (b-a) * self.phi

    def _next_b (self, a, b) -> float : 
        return b - (b-a) * self.phi
        
    def _next_a (self, a, b) -> float: 
        return (b-a)* self.phi + a

    def optimize (self, func_callback, lower_bound : float, upper_bound : float) -> float :
        a , b = lower_bound, upper_bound
        x1,x2 = self._first(a,b)
        # print(f"First iteration : {x1},{x2}")

        while ( abs(x2 - x1) > EPS) : 
            print (f"Search space is {a},{b}")
            # print(f"{func_callback(x1)} and {func_callback(x2)}")
            if (func_callback (x1) < func_callback(x2)): 
                b =  x2
                x1, x2 = self._next_b(a, b), x1
            else : 
                a = x1
                x1, x2 = x2, self._next_a(a, b)



        print("search done")
        
        return (a + b)/2


if __name__ == "__main__" :
    func = lambda x : x**4  - 14 * x **3 + 60 * x** 2 - 70 * x
    low_bound = 0
    up_bound = 2

    gs = GoldenSearch()
    soln = gs.optimize(func, low_bound, up_bound)

    print(soln)

    pass
