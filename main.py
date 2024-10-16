import numpy as np
from utils.base import Function
from utils.first_order import GradientDecent
from utils.functions import RosenBrock

if __name__ == "__main__":
    f = lambda x : x[0]**2 + x[0]*x[1] + x[1]**2 
    g = lambda x : np.array(
            [2*x[0] + x[1], x[0] + 2 * x[1]]
            )

    optim = GradientDecent(0.1)
    myfunc = Function(f, g) 
    x_0 = np.array([4,2])
    
    soln = myfunc.optimize(x_0, optim)

    print(soln)

    soln = RosenBrock.optimize(x_0, optim)

    print(soln)
