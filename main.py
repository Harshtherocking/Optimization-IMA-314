import numpy as np
from utils.base import Function
from utils.first_order import GradientDecent
from utils.functions import Rastrigin, RosenBrock, Ackley
from utils.line_search import GoldenSearch

if __name__ == "__main__":
    x = np.array([2,2])

    gs = GoldenSearch()
    gd = GradientDecent(alpha_optim=gs)

    Ackley.plot(x_val=(-10,10), y_val=(-10,10), num_points= 105)
    print(Ackley.optimize(x, optim=gd))
