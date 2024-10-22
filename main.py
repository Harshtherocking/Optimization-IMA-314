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
    x = np.array([5,7])

    adam = Adam()
    print(Rastrigin.optimize(x, optim=adam, is_plot= True))
    print(Ackley.optimize(x, optim= adam, is_plot= True))
    print(RosenBrock.optimize(x,optim=adam, is_plot= True))
    print(Trid.optimize(x, optim = adam, is_plot= True))
