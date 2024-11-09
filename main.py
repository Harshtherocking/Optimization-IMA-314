# Driver Code for the Optimization Algorithms

# Authors:
#   Ankur Majumdar (2022BCD0046)
#   Harsh Vardhan Singh Chauhan (2022BCD0044)

import numpy as np

from algorithms.base import Function
from algorithms.regression import LinearRegression, LogisticRegression
from algorithms.first_order import (
    GradientDescent,
    MomentumGradientDescent,
    NesterovAcceleratedGradientDescent,
    Adagrad,
    RMSProp,
    Adam,
    Subgradient,
)
from algorithms.second_order import NewtonMethod, BFGS, DFP, ConjugateGradient
from algorithms.line_search import GoldenSearch, Backtracking

if __name__ == "__main__":

    # Step 1: Create a Suitable Dataset consisting of X and Y (linear to be Plottable as mentioned in the Sample below)
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    Y = np.array(list(map(lambda x: x, [9, 11, 4, 5, 16, 17, 19, 8, 13, 20])))

    # Step 2: Choose an ML Algorithm based on your Use Case
    # Available ML Algorithms: LinearRegression, LogisticRegression

    # Note: If you are using Logistic Regression, kindly change the lambda function in the `Y` above to be binary (0 or 1)
    #       as the Logistic Regression requires binary output.
    #       Like it might be: Y = np.array(list(map(lambda x: x % 2, [9, 11, 4, 5, 16, 17, 19, 8, 13, 20])))

    # Note: You can also Run the Optimizers independently but you will have to define the Function, its Gradient and Hessian.

    # Step 3: Choose an Optimizer algorithm as optim Parameter of the ML Algorithm Constructor as shown below...
    # Available First Order Optimizers: GradientDescent, MomentumGradientDescent, NesterovAcceleratedGradientDescent, Adagrad, RMSProp, Adam, Subgradient
    # Available Second Order Optimizers: NewtonMethod, BFGS, DFP, ConjugateGradient

    # Step 4: (Optional) Choose an Optimizer for alpha (Learning Rate) as alpha_optim Parameter of the Optimizer Constructor as shown below...
    # Available Line Search Methods: GoldenSection, Backtracking with isArmijo=True | isArmijo=False

    lr = LinearRegression(
        optim=NesterovAcceleratedGradientDescent(
            alpha_optim=Backtracking(isArmijo=True)
        )
    )

    # Step 5: Train the ML Algorithm with the chosen Optimizer as Shown below...

    # Note: If you are using Stochastic Batch Picking, kindly set is_stochastic=True and give batch_size.
    # Always set is_plot=True to see the Trajectory of the Optimizer.

    lr.train(X_train=X, Y_train=Y, is_plot=True, is_stochastic=True, batch_size=1)
