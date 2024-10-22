import numpy as np
from numpy import ndarray
from utils.base import Optim, Function

EPSILON = 0.0001

class GradientDescent (Optim): 
    '''
    Gradient Descent - description
    '''
    def __init__ (self, damp_factor: float = 0.01, alpha_optim : None | Optim  = None) -> None: 
        self.damp_factor = damp_factor
        self.alpha_optim = alpha_optim
        return 

    def _reset (self) -> None : 
        self.alpha = 0.01
        return

    def _next (self, x: ndarray, func_callback, grad_func_callback) -> ndarray : 

        if isinstance(self.alpha_optim, Optim) : 

            self.alpha = self.alpha_optim.optimize(
                    func_callback= lambda alpha : func_callback(x - alpha * grad_func_callback(x)),
                    lower_bound = 0, upper_bound = 1
                    ) 

        # print(f"Alpha : {self.alpha}")
        return x - self.alpha * grad_func_callback(x)

    def optimize (self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback, grad_mod_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        plot_points : list[ndarray] = [x]

        while (grad_mod_callback(x) > EPSILON): 
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset
        if is_plot : 
            return x, plot_points
        return x
