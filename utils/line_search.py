import numpy as np
from numpy import ndarray

from utils.base import Optim

EPS = 0.3

class GoldenSearch(Optim):
    def __init__(self, phi: float = 0.68103) -> None:
        self.phi = phi 

    def _first(self, a, b) -> tuple[float, float]:
        x1 = b - (b - a) * self.phi
        x2 = a + (b - a) * self.phi
        return x1, x2

    def _next_b(self, a, b) -> float:
        return b - (b - a) * self.phi

    def _next_a(self, a, b) -> float:
        return a + (b - a) * self.phi

    def optimize(self, func_callback, lower_bound: float, upper_bound: float) -> float:
        a, b = lower_bound, upper_bound
        x1, x2 = self._first(a, b)

        while abs(b - a) > EPS:
            if func_callback(x1) < func_callback(x2):
                b = x2
                x2 = x1
                x1 = self._next_b(a, b)
            else:
                a = x1
                x1 = x2
                x2 = self._next_a(a, b)
        
        return (a + b) / 2


if __name__ == "__main__":
    func = lambda x: x**4 - 14 * x**3 + 60 * x**2 - 70 * x
    low_bound = 0
    up_bound = 2

    gs = GoldenSearch()
    soln = gs.optimize(func, low_bound, up_bound)

    print("Optimal solution:", soln)
