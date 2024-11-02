import numpy as np

from utils.base import Function
from utils.functions import RosenBrock
from utils.regression import LinearRegression, LogisticRegression

from utils.line_search import GoldenSearch, Backtracking

from utils.first_order import GradientDescent, Adam, Adagrad
from utils.second_order import NewtonMethod, BFGS, DFP, ConjugateGradient

if __name__ == "__main__":

    X = np.array(
        [
            [0.1, 0.2],
            [0.4, 0.8],
            [0.5, 1.0],
            [0.7, 1.4],
            [0.9, 1.8],
            [1.2, 2.4],
            [1.5, 3.0],
            [1.7, 3.4],
            [2.0, 4.0],
            [2.3, 4.6],
            [0.25, 0.5],
            [0.45, 0.9],
            [0.55, 1.1],
            [0.75, 1.5],
            [0.95, 1.9],
            [1.3, 2.6],
            [1.45, 2.9],
            [1.65, 3.3],
            [1.85, 3.7],
            [2.15, 4.3],
            [0.15, 0.3],
            [0.35, 0.7],
            [0.65, 1.3],
            [1.1, 2.2],
            [1.4, 2.8],
            [1.6, 3.2],
            [1.8, 3.6],
            [2.1, 4.2],
        ]
    )

    # Generate binary target variable Y
    Y = np.array(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
    )

    gd = GradientDescent(alpha_optim=Backtracking())

    lr = LogisticRegression(optim=gd)

    lr.train(X, Y, is_plot=True)

    print(lr(np.array([[1, 2]])))

    myfunc = Function(func=lambda x: 0.5 * x[0] ** 2 + 10 * x[1] ** 2, name="sample")

    initial = np.array([3, 3])

    print(myfunc.optimize(initial, optim=ConjugateGradient(), is_plot=True))
