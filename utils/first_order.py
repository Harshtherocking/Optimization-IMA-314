import numpy as np
from numpy import ndarray
from utils.base import Optim, Function

EPSILON = 0.001

class GradientDecent (Optim): 
    '''
    Gradient Decent - description
    '''
    def __init__ (self, alpha: float = 0.01, alpha_optim : None | Optim  = None) -> None: 
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        return 
    
    def _reset (self) -> None : 
        self.alpha = 0.01
        return

    def _next (self, x: ndarray, func_callback, grad_func_callback) -> ndarray: 

        if isinstance(self.alpha_optim, Optim) : 

            self.alpha = self.alpha_optim.optimize(
                    func_callback= lambda alpha : func_callback(x - alpha * grad_func_callback(x)),
                    lower_bound = 0, upper_bound = 1
                    ) 

        # print(f"Alpha : {self.alpha}")
        return x - self.alpha * grad_func_callback(x)
    
    def optimize (self, x: ndarray, func_callback, grad_func_callback, grad_mod_callback) -> ndarray: 

        while (grad_mod_callback(x) > EPSILON): 
            x = self._next(x, func_callback, grad_func_callback)

        return x


if __name__ == "__main__" : 
    f = lambda x : x[0]**2 + x[1] **2
    g = lambda x : np.array([2*x[0], 2*x[1]])

    func = Function(f,g, "func")
    x  = np.array([3,8])

    gd = GradientDecent()

    soln  = func.optimize(x, gd)
    
    print(soln)
