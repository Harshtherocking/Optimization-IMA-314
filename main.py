import numpy as np
from utils.base import Function

from utils.first_order import (
        GradientDecent,
        NesterovAcceleratedGradientDescent,
        Adagrad,
        RMSProp,
        Adam
        ) 

from utils.functions import (
         Rastrigin,
         RosenBrock,
         Ackley,
         Trid
        )
from utils.line_search import GoldenSearch

if __name__ == "__main__":
    # declare a funciton 
    f = lambda x : 0.5 * x.T @ np.array([[1,0],[0,2]]) @ x - x.T @ np.array([1,1]) + 7

    x = np.array([2,1])

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
    soln = sampleFunc.optimize (x, optim= dfp, is_plot = True)

    print(f"Optimize x : {soln}")


