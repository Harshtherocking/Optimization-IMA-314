import numpy as np
from numpy import ndarray
from utils.base import Optim, Function

EPSILON = 0.0001

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

    def _next (self, x: ndarray, func_callback, grad_func_callback) -> ndarray : 

        if isinstance(self.alpha_optim, Optim) : 

            self.alpha = self.alpha_optim.optimize(
                    func_callback= lambda alpha : func_callback(x - alpha * grad_func_callback(x)),
                    lower_bound = 0, upper_bound = 1
                    ) 

        # print(f"Alpha : {self.alpha}")
        return x - self.alpha * grad_func_callback(x)

    def optimize (self, x: ndarray, func_callback, grad_func_callback, grad_mod_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        plot_points : list[ndarray] = [x]

        while (grad_mod_callback(x) > EPSILON): 
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset
        if is_plot : 
            return x, plot_points
        return x



class NesterovAcceleratedGradientDescent (Optim):
    '''
    NGD description
    '''
    def __init__ (self, alpha : float = 0.01, momemtum_coff : float = 0.75) -> None : 
        self.alpha = alpha 
        self.momemtum_coff  = momemtum_coff
        self.momentum : ndarray | None = None;
        return

    def _reset (self) -> None : 
        self.alpha = 0.01 
        self.momemtum_coff  = 0.75
        self.momentum : ndarray | None = None;
        return


    def _next (self, x : ndarray, grad_mod_callback) -> ndarray :
        assert (isinstance(self.momentum, ndarray)), "initial momentum not defined"

        x_look_ahead = x - self.momemtum_coff * self.momentum
        self.momentum = self.momemtum_coff * self.momentum  + self.alpha * grad_mod_callback(x_look_ahead)

        return x - self.momentum


    def optimize (self, x: ndarray, func_callback, grad_func_callback, grad_mod_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        self.momentum = np.zeros(x.shape)
        plot_points : list[ndarray] = [x]

        while (grad_mod_callback(x) > EPSILON): 
            x = self._next(x, grad_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset
        if is_plot : 
            return x, plot_points
        return x


class Adagrad (Optim):
    '''
    Adagrad description
    '''
    def __init__ (self, alpha : float = 0.01) -> None : 
        self.alpha = alpha 
        self.sq_grad_acc: ndarray | None = None;
        return

    def _reset (self) -> None : 
        self.sq_grad_acc: ndarray | None = None;
        return


    def _next (self, x : ndarray, grad_mod_callback) -> ndarray :
        assert (isinstance(self.sq_grad_acc, ndarray)), "initial square gradient accumulation not defined"
        self.sq_grad_acc += np.square(grad_mod_callback(x))

        assert (isinstance(self.sq_grad_acc, ndarray)), "problem in accumulation"
        return x - self.alpha / np.sqrt(self.sq_grad_acc + EPSILON) * grad_mod_callback(x)


    def optimize (self, x: ndarray, func_callback, grad_func_callback, grad_mod_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        self.sq_grad_acc = np.zeros(x.shape)
        plot_points : list[ndarray] = [x]

        while (grad_mod_callback(x) > EPSILON): 
            x = self._next(x, grad_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset
        if is_plot : 
            return x, plot_points
        return x



class RMSProp(Optim):
    '''
    RMSProp description
    '''
    def __init__ (self, alpha : float = 0.01, beta : float = 0.9) -> None : 
        self.alpha = alpha 
        self.beta = beta
        self.sq_grad_acc: ndarray | None = None;
        return

    def _reset (self) -> None : 
        self.sq_grad_acc: ndarray | None = None;
        return


    def _next (self, x : ndarray, grad_mod_callback) -> ndarray :
        assert (isinstance(self.sq_grad_acc, ndarray)), "initial square gradient accumulation not defined"
        self.sq_grad_acc = self.beta * self.sq_grad_acc + (1 - self.beta) * np.square(grad_mod_callback(x))

        assert (isinstance(self.sq_grad_acc, ndarray)), "problem in accumulation"
        return x - self.alpha / np.sqrt(self.sq_grad_acc + EPSILON) * grad_mod_callback(x)


    def optimize (self, x: ndarray, func_callback, grad_func_callback, grad_mod_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        self.sq_grad_acc = np.zeros(x.shape)
        plot_points : list[ndarray] = [x]

        while (grad_mod_callback(x) > EPSILON): 
            x = self._next(x, grad_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset
        if is_plot : 
            return x, plot_points
        return x


class Adam (Optim):
    '''
    Adam description
    '''
    def __init__ (self, alpha : float = 0.01, beta_1 : float = 0.9, beta_2 : float = 0.99) -> None : 
        self.alpha = alpha 
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.first_order_acc : ndarray | None = None
        self.second_order_acc : ndarray | None = None
        return

    def _reset (self) -> None : 
        self.first_order_acc : ndarray | None = None
        self.second_order_acc : ndarray | None = None
        return


    def _next (self, x : ndarray, grad_mod_callback) -> ndarray :
        assert (isinstance(self.first_order_acc, ndarray)), "initial first order accumulation not defined"
        assert (isinstance(self.second_order_acc, ndarray)), "initial second order accumulation not defined"

        self.first_order_acc = self.beta_1 * self.first_order_acc + (1 - self.beta_1) * grad_mod_callback(x)
        self.second_order_acc = self.beta_2 * self.second_order_acc + (1 - self.beta_2) * np.square(grad_mod_callback(x)) 

        assert (isinstance(self.first_order_acc, ndarray)), "problem in first order accumulation"
        assert (isinstance(self.second_order_acc, ndarray)), "problem in second order accumulation"

        first_order_corrected = self.first_order_acc / (1-self.beta_1)
        second_order_corrected = self.second_order_acc / (1-self.beta_2)

        assert (isinstance(first_order_corrected, ndarray)), "problem in first order correction"
        assert (isinstance(second_order_corrected, ndarray)), "problem in second order correction"

        return x - self.alpha / np.sqrt(second_order_corrected + EPSILON) * first_order_corrected


    def optimize (self, x: ndarray, func_callback, grad_func_callback, grad_mod_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        self.first_order_acc = np.zeros(x.shape)
        self.second_order_acc = np.zeros(x.shape)
        plot_points : list[ndarray] = [x]

        while (grad_mod_callback(x) > EPSILON): 
            x = self._next(x, grad_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset
        if is_plot : 
            return x, plot_points
        return x


class Subgradient (Optim): 
    '''
    Subgradient - description
    '''
    def __init__ (self, alpha: float = 0.01, alpha_optim : None | Optim  = None) -> None: 
        self.alpha = alpha
        self.f_best : float | None = None
        self.K = 100
        return 

    def _reset (self) -> None : 
        self.alpha = 0.01
        self.K
        return

    def _next (self, x: ndarray, func_callback, grad_func_callback) -> ndarray : 
        x_new = x - self.alpha * grad_func_callback(x)

        if (func_callback(x_new) < self.f_best) : 
            self.f_best = func_callback(x_new)
            print(f"F_Best : {self.f_best}")
            return x_new

        self.K -= 1
        return x

    def optimize (self, x: ndarray, func_callback, grad_func_callback, grad_mod_callback, is_plot : bool = False) -> ndarray | tuple[ndarray,list[ndarray]]: 
        plot_points : list[ndarray] = [x]
        self.f_best = func_callback(x)

        while (grad_mod_callback(x) > EPSILON and self.K > 0): 
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot :
                plot_points.append(x)

        self._reset
        if is_plot : 
            return x, plot_points
        return x



if __name__ == "__main__" : 
    f = lambda x : x[0]**2 + x[1] **2
    g = lambda x : np.array([2*x[0], 2*x[1]])

    func = Function(f,g, "func")
    x  = np.array([3,8])

    gd = GradientDecent()

    soln  = func.optimize(x, gd)

    print(soln)
