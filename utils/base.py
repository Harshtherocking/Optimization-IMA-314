import numpy as np
from numpy import ndarray
from abc import abstractmethod, ABC 

class Optim (ABC): 
    '''
    Base class for optimizers
    '''
    @abstractmethod
    def optimize (self, x : ndarray, func_callback, grad_func_callback, grad_mod_callback) -> ndarray :
        pass


class Function : 
    '''
    Base class for functions
    func : function which should return : ndarray() object type
    grad : gradient function returning ndarray() obeject type
    '''
    def __init__ (self, func , grad_func ,  name : str = "myFunc" ) -> None :
        self.func = func
        self.grad_func = grad_func
        self.name = name
        return None

    def __call__ (self, x : ndarray) -> ndarray :  
        return self.func(x)
    
    def __repr__(self) -> str:
        return self.name

    def grad (self, x : ndarray) -> ndarray : 
        return self.grad_func(x)

    def grad_mod (self, x : ndarray) -> float : 
        l2_norm = np.sum(np.square(self.grad(x)))
        return np.sqrt(l2_norm)

    def optimize (self, initial_val : ndarray, optim: Optim) -> ndarray : 
        return  optim.optimize(initial_val, self.__call__, self.grad, self.grad_mod)
