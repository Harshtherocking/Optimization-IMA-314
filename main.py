import numpy as np
from utils.base import Function

from utils.first_order import (
        GradientDecent,
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
from utils.line_search import GoldenSearch



if __name__ == "__main__":
    x = np.array([5,7])

    # print(PiecewiseLinear.grad( np.array([1,1]) ))
    # print(PiecewiseLinear.grad( np.array([-1,1]) ))
    # print(PiecewiseLinear.grad( np.array([1,-1]) ))
    # print(PiecewiseLinear.grad( np.array([-1,-1]) ))
    # print(PiecewiseLinear.grad( np.array([0,1]) ))
    # print(PiecewiseLinear.grad( np.array([1,0]) ))
    # print(PiecewiseLinear.grad( np.array([-1,0]) ))
    # print(PiecewiseLinear.grad( np.array([0,-1]) ))
    # print(PiecewiseLinear.grad( np.array([0,0]) ))

    # PiecewiseLinear.plot()

    sg = Subgradient()
    print(PiecewiseLinear.optimize(x, optim =sg, is_plot = True))


    
    # adam = Adam()
    # print(Rastrigin.optimize(x, optim=adam, is_plot= True))
    # print(Ackley.optimize(x, optim= adam, is_plot= True))
    # print(RosenBrock.optimize(x,optim=adam, is_plot= True))
    # print(Trid.optimize(x, optim = adam, is_plot= True))
