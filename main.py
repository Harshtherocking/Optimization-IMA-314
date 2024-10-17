import numpy as np
from utils.base import Function
from utils.first_order import GradientDecent
from utils.functions import RosenBrock

if __name__ == "__main__":
    x = np.array([2,2])

    gd = GradientDecent()

    RosenBrock.plot()
    print(RosenBrock.optimize(x, optim=gd))