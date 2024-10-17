import numpy as np
from numpy import ndarray
from .base import Optim, Function

EPSILON = 0.001

class GradientDecent (Optim): 
    '''
    Gradient Decent - description
    '''
    def __init__ (self, alpha: float = 0.01) -> None: 
        self.alpha = alpha
        return 

    def _next (self, x: ndarray, grad_func) -> ndarray: 
        return x - self.alpha * grad_func(x)
    
    def optimize (self, x: ndarray, func_callback, grad_func_callback, grad_mod_callback) -> ndarray: 
        while (grad_mod_callback(x) > EPSILON): 
            x = self._next(x, grad_func_callback)
        return x


if __name__ == "__main__" : 
    f = lambda x : x[0]**2 + x[1] **2
    g = lambda x : np.array([2*x[0], 2*x[1]])

    func = Function(f,g, "func")
    x  = np.array([3,8])

    gd = GradientDecent()

    soln  = func.optimize(x, gd)
    
    print(soln)
