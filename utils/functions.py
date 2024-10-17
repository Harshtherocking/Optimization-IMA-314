import numpy as np
from .base import Function

A = 1 
B = 10

RosenBrock = Function (
        func = lambda x : (A-x[0])**2  + B*(x[1] - x[0] **2) **2, 
        grad_func = lambda x : np.array ([
            -2* (A-x[0]) - 4*B*x[0]*(x[1] - x[0]**2),
            2*B* (x[1] - x[0]**2)
            ]),
        name = "rosenbrock"
        )
