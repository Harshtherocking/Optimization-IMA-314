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
        NewtonMethod,
        DFP, 
        BFGS
        )

from utils.line_search import GoldenSearch



if __name__ == "__main__":
    # declare a funciton 
    f = lambda x :  8 * x[0] + x[1] ** 2 + 2 * x[0] ** 2 

    # pass into Funtion object
    sampleFunc = Function (f, name = "samplefunc")

    # plot the function
    sampleFunc.plot()

    # define optimization algorithms
    gs = GoldenSearch () 
    nm = NewtonMethod(alpha_optim= gs)
    dfp = DFP(alpha_optim= gs)
    bfgs = BFGS(alpha_optim =gs)

    # optimize and plot trajectory
    x = np.array([2,1])
    soln = sampleFunc.optimize (x, optim= bfgs, is_plot = True)

    print(f"Optimize x : {soln}")


