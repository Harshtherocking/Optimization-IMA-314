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

    def optimize(self, x, Func, func_callback, grad_func_callback, lower_bound: float, upper_bound: float) -> float:
        # defining target function to optimize
        Func = Func if Func else lambda alpha : func_callback(x - alpha * grad_func_callback(x))

        a, b = lower_bound, upper_bound
        x1, x2 = self._first(a, b)

        while abs(b - a) > EPS:
            if Func(x1) < Func(x2):
                b = x2
                x2 = x1
                x1 = self._next_b(a, b)
            else:
                a = x1
                x1 = x2
                x2 = self._next_a(a, b)
        
        return (a + b) / 2



class Backtracking (Optim):
    def __init__(self, beta: float = 0.5, delta: float = 0.1, isArmijo: bool = True) -> None:
        # diminishing factor
        self.beta = beta
        # controlling factor
        self.delta = delta

        self.isArmijo = isArmijo

    def _check_condition(self,alpha, Func) -> bool:
        return Func(alpha)

    def optimize(self, x, Func, func_callback, grad_func_callback, lower_bound: float, upper_bound: float) -> float:
        # defining target function to optimize
        Func = lambda alpha : func_callback(x - alpha * grad_func_callback(x)) > func_callback(x) - self.delta * alpha * (grad_func_callback(x).T @ grad_func_callback(x)) if self.isArmijo else lambda alpha : func_callback(x - alpha * grad_func_callback(x)) > func_callback(x) 

        alpha = upper_bound

        while self._check_condition(alpha, Func) and alpha > lower_bound:
            alpha *= self.beta
        return alpha


