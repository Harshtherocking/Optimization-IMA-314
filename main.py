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
from utils.line_search import GoldenSearch



if __name__ == "__main__":
    x = np.array([5,7])


    adam = Adam()
    sg = Subgradient()
    f  = lambda x : x[0] ** 2 + x[0] * x[1] + x[1] ** 2
    myfunc = Function(f)

    print(PiecewiseLinear.optimize(x, optim= sg , is_plot= True))




    # Rastrigin.plot([x])
    # print(Rastrigin(x), Rastrigin.grad(x), Rastrigin.hessian(x), sep = "\n")
    

    # sg = Subgradient()
    # print(PiecewiseLinear.optimize(x, optim =sg, is_plot = True))


    
    # adam = Adam()
    # print(Rastrigin.optimize(x, optim=adam, is_plot= True))
    # print(Ackley.optimize(x, optim= adam, is_plot= True))
    # print(RosenBrock.optimize(x,optim=adam, is_plot= True))
    # print(Trid.optimize(x, optim = adam, is_plot= True))
