import numpy as np
from numpy import ndarray
from numpy.linalg import inv, eig
from utils.base import Optim, Function

EPSILON = 0.0001

class DampedNewton (Optim): 
    '''
    Levenberg-Marquardt Modification - description
    '''
    def __init__ (self, alpha : float = 0.01, damp_factor: float = 0.1, alpha_optim : None | Optim  = None) -> None: 
        self.damp_factor = damp_factor
        return 

    def _reset (self) -> None : 
        self.damp_factor = 0.1
        self.num_iter = 0
        return

    def _next (self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback) -> ndarray : 
        # checking for positive definite hessian 
        if (eig(hessian_func_callback(x))[0].all() > 0):
            return x - inv(hessian_func_callback(x)) @ grad_func_callback(x)

        # if not a positive definite hessian
        else:   
            # Levenberg-Marquardt modification
            delta = inv(hessian_func_callback(x) + self.damp_factor * np.identity(x.shape[0])) @ grad_func_callback(x)

            # checking if delta is in the decent direction 
            while (func_callback(x) < func_callback(x - delta)) : 
                self.damp_factor += 0.1
                delta = inv(hessian_func_callback(x) + self.damp_factor * np.identity(x.shape[0])) @ grad_func_callback(x)

            return x - delta

    def optimize (self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback, grad_mod_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        plot_points : list[ndarray] = [x]

        while (grad_mod_callback(x) > EPSILON): 
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback, hessian_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset
        if is_plot : 
            return x, plot_points
        return x
