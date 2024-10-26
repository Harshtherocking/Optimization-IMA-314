import numpy as np
from utils.base import Function
from utils.regression import LinearRegression

from utils.first_order import GradientDescent, Adam, Adagrad

from utils.functions import RosenBrock

from utils.second_order import NewtonMethod

from utils.line_search import GoldenSearch


if __name__ == "__main__":
    X = np.array([
        [1,2],
        [2,3],
        [3,4],
        ])

    Y = np.array([3,5,7])

    gd = GradientDescent(alpha_optim = GoldenSearch())

    lr = LinearRegression(optim = gd)
    lr.train(X_train= X, Y_train= Y, is_plot= True)

    x = np.array([
        [1],
        [4]
        ])



