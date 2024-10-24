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
        DFP
        )

from utils.line_search import GoldenSearch



if __name__ == "__main__":
    # declare a funciton 
    f = lambda x : x[0] ** 2 + 0.5 * x[1] ** 2

    # pass into Funtion object
    sampleFunc = Function (f, name = "samplefunc")

    # plot the function
    sampleFunc.plot()

    # define optimization algorithms
    gs = GoldenSearch () 
    nm = NewtonMethod(alpha_optim= gs)
    dfp = DFP(alpha_optim= gs)

    # optimize and plot trajectory
    x = np.array([7,10])
    soln = sampleFunc.optimize (x, optim= dfp, is_plot = True)

    print(f"Optimize x : {soln}")


