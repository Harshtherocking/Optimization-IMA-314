import numpy as np
from numpy import ndarray
from base import Optim

EPSILON = 0.001

class GradientDecent (Optim): 
    '''
    Gradient Decent - description
    '''
    def __init__ (self, alpha: float) -> None: 
        self.alpha = alpha
        return 
    def _next (self, x: ndarray, grad_func) -> ndarray: 
        return x - self.alpha * grad_func(x)
    
    def optimize (self, x: ndarray, func_callback, grad_func_callback, grad_mod_callback) -> ndarray: 
        while (grad_mod_callback(x) > EPSILON): 
            x = self._next(x, grad_func_callback)
        return x
