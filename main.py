import numpy as np
from utils.base import Function

from utils.first_order import (
        GradientDescent,
        NesterovAcceleratedGradientDescent,
        Adagrad,
        RMSProp,
        Adam, 
        Subgradient
        ) 

from utils.functions import (
         Rastrigin,
         RosenBrock,
         Ackley,
         Trid, 
         PiecewiseLinear
        )

from utils.second_order import (
        DampedNewton
        )

from utils.line_search import GoldenSearch



if __name__ == "__main__":
    # declare a funciton 
    f = lambda x : x[0] ** 2 + 0.5 * x[1] ** 2

    # pass into Funtion object
    sampleFunc = Function (f, name = "samplefunc")

    # plot the function
    sampleFunc.plot()

    # get value for a specific point
    x = np.array([5,2])

    func_val = sampleFunc(x)
    grad_val = sampleFunc.grad(x)
    hess_val = sampleFunc.hessian(x)

    print(f"At {x}\nF(x) = {func_val}\nG(x) = {grad_val}\nH(x) = {hess_val}")
    
    # define optimization algorithms
    gs = GoldenSearch () 
    gd = GradientDescent (alpha = 0.01, alpha_optim = gs)

    # optimize and plot trajectory
    x = np.array([7,10])
    soln = sampleFunc.optimize (x, optim= gd, is_plot = True)
    print(f"Optimize x : {soln}")


