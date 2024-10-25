import numpy as np
from numpy import ndarray
from numpy.linalg import inv, eig
from utils.base import Optim, Function

EPSILON = 0.0001

class NewtonMethod (Optim): 
    '''
    Levenberg-Marquardt Modification - description
    '''
    def __init__ (self, alpha : float = 0.01, damp_factor: float = 0.1, alpha_optim : None | Optim  = None) -> None: 
        self.alpha = alpha
        self.damp_factor = damp_factor
        self.alpha_optim = alpha_optim
        return 

    def _reset (self) -> None : 
        self.alpha = 0.01
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

            # ensuring the resultant damp factor is bigger than abs 
            # (minimum of  eigen values of hessian) 
            # so that resultant hessian is positive hessian 
            delta = inv(hessian_func_callback(x) + ( self.damp_factor + abs( min(eig(hessian_func_callback(x))[0]) ) ) * np.identity(x.shape[0])) @ grad_func_callback(x)

            #optimize alpha 
            if isinstance(self.alpha_optim, Optim) : 

                self.alpha = self.alpha_optim.optimize(
                        func_callback= lambda alpha : func_callback(x - alpha * delta),
                        lower_bound = 0, upper_bound = 1
                        ) 

            return x - self.alpha * delta

    def optimize (self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        plot_points : list[ndarray] = [x]

        while (np.linalg.norm(grad_func_callback(x)) > EPSILON): 
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback, hessian_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset
        if is_plot : 
            return x, plot_points
        return x
    



# Quasi Newton Algorithms
'''
Quasi Newton Methods use Hessian approximation instead of real Hessian
'''

class DFP (Optim): 
    '''
    Davidon Fletcher Powell - description
    '''
    def __init__ (self, alpha : float = 0.01,  alpha_optim : None | Optim  = None) -> None: 
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        self.hessian_k : ndarray | None = None
        return 

    def _reset (self) -> None : 
        self.alpha = 0.01
        self.num_iter = 0
        self.hessian_k : ndarray | None = None
        return


    def _next (self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback) -> ndarray : 
        
        # first step, take real hessian not approximation 
        # this wont run for subsequent iterations
        if (not isinstance(self.hessian_k, ndarray)) : 
            self.hessian_k = hessian_func_callback(x)

        assert(isinstance(self.hessian_k, ndarray)), "hessian not defined correctly"
        assert(eig(self.hessian_k)[0].all() > 0), "Not positive definite hessian"

        direction = - self.hessian_k @ grad_func_callback(x) 

        # optimize search direction alpha 
        if isinstance(self.alpha_optim, Optim) : 

            self.alpha = self.alpha_optim.optimize(
                    func_callback= lambda alpha : func_callback(x + alpha * direction),
                    lower_bound = 0, upper_bound = 1
                    ) 

        x_new = x + self.alpha * direction

        del_x = self.alpha * direction
        del_grad = grad_func_callback(x_new) - grad_func_callback(x)
        
        # calculating hessian for next iteration
        self.hessian_k = self.hessian_k + (del_x @ del_x.T) / (del_x.T @ del_grad) - ((self.hessian_k @ del_grad) @ (self.hessian_k @ del_grad).T) / del_grad.T @ self.hessian_k @ del_grad 

        return x_new

 
    def optimize (self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        plot_points : list[ndarray] = [x]

        while (np.linalg.norm(grad_func_callback(x)) > EPSILON): 
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback, hessian_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset

        if is_plot : 
            return x, plot_points
        return x




class BFGS (Optim): 
    '''
    Broyden Fletcher Goldfarb Shanno - description
    '''
    def __init__ (self, alpha : float = 0.01,  alpha_optim : None | Optim  = None) -> None: 
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        self.hessian_k : ndarray | None = None
        return 

    def _reset (self) -> None : 
        self.alpha = 0.01
        self.num_iter = 0
        self.hessian_k : ndarray | None = None
        return


    def _next (self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback) -> ndarray : 
        
        # first step, take real hessian not approximation 
        # this wont run for subsequent iterations
        if (not isinstance(self.hessian_k, ndarray)) : 
            self.hessian_k = hessian_func_callback(x)

        assert(isinstance(self.hessian_k, ndarray)), "hessian not defined correctly"
        assert(eig(self.hessian_k)[0].all() > 0), "Not positive definite hessian"

        direction = - self.hessian_k @ grad_func_callback(x) 

        # optimize search direction alpha 
        if isinstance(self.alpha_optim, Optim) : 

            self.alpha = self.alpha_optim.optimize(
                    func_callback= lambda alpha : func_callback(x + alpha * direction),
                    lower_bound = 0, upper_bound = 1
                    ) 

        x_new = x + self.alpha * direction

        del_x = self.alpha * direction
        del_grad = grad_func_callback(x_new) - grad_func_callback(x)
        
        # calculating hessian for next iteration
        self.hessian_k = inv(inv(self.hessian_k) + (del_grad @ del_grad.T) / (del_grad.T @ del_x) - ((inv(self.hessian_k) @ del_x) @ (inv(self.hessian_k) @ del_x).T) / del_x.T @ inv(self.hessian_k) @ del_x )

        return x_new

 
    def optimize (self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        plot_points : list[ndarray] = [x]

        while (np.linalg.norm(grad_func_callback(x)) > EPSILON): 
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback, hessian_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset

        if is_plot : 
            return x, plot_points
        return x
