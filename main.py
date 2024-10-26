import numpy as np
from utils.base import Function
from utils.regression import LinearRegression

from utils.first_order import GradientDescent, Adam, Adagrad

from utils.functions import RosenBrock

from utils.second_order import NewtonMethod, BFGS, DFP

from utils.line_search import GoldenSearch, Backtracking

if __name__ == "__main__":
    X = np.array(
        [
            [1],
            [2],
            [3],
        ]
    )

    Y = np.array([3, 5, 7])

    gd = GradientDescent(alpha_optim=Backtracking(isArmijo=True))

    lr = LinearRegression(optim=BFGS())
    lr.train(X_train=X, Y_train=Y, is_plot=True)

    x = np.array([[1], [4]])
