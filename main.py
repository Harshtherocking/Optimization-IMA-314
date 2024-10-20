import numpy as np
from utils.base import Function

from utils.first_order import (
        GradientDecent,
        NesterovAcceleratedGradientDescent,
        Adagrad,
        RMSProp,
        Adam
        ) 

from utils.functions import Rastrigin, RosenBrock, Ackley
from utils.line_search import GoldenSearch

if __name__ == "__main__":
    x = np.array([2,2])

    gs = GoldenSearch()
    gd = GradientDecent(alpha_optim=gs)
    ngd = NesterovAcceleratedGradientDescent(alpha= 0.02, momemtum_coff= 0.7)

    adagrad = Adagrad(alpha= 0.05)

    rmsprop = RMSProp()

    adam = Adam()
    # Ackley.plot(x_val=(-10,10), y_val=(-10,10), num_points= 105)
    Rastrigin.plot(x_val=(-10,10), y_val=(-10,10), num_points= 105)
    
    
    # print(Rastrigin.optimize(x, optim=gd))
    # print(Rastrigin.optimize(x, optim=ngd))
    # print(Rastrigin.optimize(x, optim=adagrad))
    # print(Rastrigin.optimize(x, optim=rmsprop))
    print(Rastrigin.optimize(x, optim=adam))
