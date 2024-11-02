import numpy as np
from numpy import ndarray
from numpy.linalg import inv, eig
from algorithms.base import Optim

EPSILON = 0.0001


class NewtonMethod(Optim):
    """
    Implementation of the Newton Method for optimization (II-Order).
    - alpha: Initial step size for the update, default is 0.01.
    - damp_factor: Damping factor used in the Levenberg-Marquardt modification to ensure a positive definite Hessian, default is 0.1.
    - alpha_optim: An optional optimization strategy that can be used to adjust the step size, default is None.

    Key Methods:
    - _next(x, func_callback, grad_func_callback, hessian_func_callback):
        Computes the next point in the optimization process based on the current point x.
        It checks if the Hessian matrix is positive definite. If it is, it directly applies the Newton update.
        If the Hessian is not positive definite, it uses the Levenberg-Marquardt modification to ensure a valid update by adding a scaled identity matrix to the Hessian.
    - optimize(x, func_callback, grad_func_callback, hessian_func_callback, is_plot):
        Iteratively updates the point x until convergence is achieved, defined by the norm of the gradient falling below a specified epsilon value.
        If is_plot is True, it records the points visited during the optimization process for potential visualization.

    The Newton Method is effective for finding local minima of differentiable functions, leveraging second-order derivative information to achieve faster convergence.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        damp_factor: float = 0.1,
        alpha_optim: None | Optim = None,
    ) -> None:
        self.alpha = alpha
        self.damp_factor = damp_factor
        self.alpha_optim = alpha_optim
        return

    def _reset(self) -> None:
        self.alpha = 0.01
        self.damp_factor = 0.1
        self.num_iter = 0
        return

    def _next(
        self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback
    ) -> ndarray:
        # checking for positive definite hessian
        if eig(hessian_func_callback(x))[0].all() > 0:
            return x - inv(hessian_func_callback(x)) @ grad_func_callback(x)

        # if not a positive definite hessian
        else:
            # Levenberg-Marquardt modification

            # ensuring the resultant damp factor is bigger than abs
            # (minimum of  eigen values of hessian)
            # so that resultant hessian is positive hessian
            delta = inv(
                hessian_func_callback(x)
                + (self.damp_factor + abs(min(eig(hessian_func_callback(x))[0])))
                * np.identity(x.shape[0])
            ) @ grad_func_callback(x)

            return x - self.alpha * delta

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback, hessian_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x


# Quasi Newton Algorithms
"""
Quasi Newton Methods use Hessian approximation instead of real Hessian due to computational complexity
"""


class DFP(Optim):
    """
    Implementation of the Davidon-Fletcher-Powell (DFP) algorithm for optimization (II-Order).

    The DFP algorithm is a quasi-Newton method that updates the inverse Hessian approximation
    at each iteration based on the difference in gradients and step sizes. This method is
    particularly useful for optimizing differentiable functions.

    Parameters:
    - alpha: Step size for the update, default is 0.01.
    - alpha_optim: An optional optimization strategy for adjusting the step size, default is None.

    Key Methods:
    - _next(x, func_callback, grad_func_callback, hessian_func_callback):
        Computes the next point in the optimization process based on the current point x.
        It starts by evaluating the Hessian at the current point if it's not already defined.
        It then computes the search direction using the negative of the product of the current
        inverse Hessian and the gradient. After calculating the new point, it updates the
        inverse Hessian approximation using the DFP formula based on the change in position
        and the change in gradient.
    - optimize(x, func_callback, grad_func_callback, hessian_func_callback, is_plot):
        Iteratively updates the point x until convergence is achieved, defined by the norm of the
        gradient falling below a specified epsilon value.
        If is_plot is True, it records the points visited during the optimization process for
        potential visualization.

    The DFP algorithm is effective for finding local minima of functions and can provide
    better convergence properties than gradient descent alone by utilizing curvature information.
    """

    def __init__(self, alpha: float = 0.01, alpha_optim: None | Optim = None) -> None:
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        self.hessian_k: ndarray | None = None
        return

    def _reset(self) -> None:
        self.alpha = 0.01
        self.num_iter = 0
        self.hessian_k: ndarray | None = None
        return

    def _next(
        self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback
    ) -> ndarray:

        # first step, take real hessian not approximation
        # this wont run for subsequent iterations
        if not isinstance(self.hessian_k, ndarray):
            self.hessian_k = hessian_func_callback(x)

        assert isinstance(self.hessian_k, ndarray), "hessian not defined correctly"
        assert eig(self.hessian_k)[0].all() > 0, "Not positive definite hessian"

        direction = -self.hessian_k @ grad_func_callback(x)

        x_new = x + self.alpha * direction

        del_x = self.alpha * direction
        del_grad = grad_func_callback(x_new) - grad_func_callback(x)

        # resizing del_x and del_grad to encounter transposing issue
        assert isinstance(del_x, ndarray)
        assert isinstance(del_grad, ndarray)

        del_x.resize((len(del_x), 1))
        del_grad.resize((len(del_grad), 1))

        # calculating hessian for next iteration
        self.hessian_k = (
            self.hessian_k
            + (del_x @ del_x.T) / (del_x.T @ del_grad)
            - ((self.hessian_k @ del_grad) @ (self.hessian_k @ del_grad).T)
            / del_grad.T
            @ self.hessian_k
            @ del_grad
        )

        return x_new

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback, hessian_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()

        if is_plot:
            return x, plot_points
        return x


class BFGS(Optim):
    """
    Implementation of the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm for optimization (II-Order).

    The BFGS algorithm is a popular quasi-Newton method that approximates the inverse Hessian
    matrix to find local minima of differentiable functions. It is known for its efficiency and
    robustness in optimizing non-linear functions, particularly in high-dimensional spaces.

    Parameters:
    - alpha: Step size for the update, default is 0.01.
    - alpha_optim: An optional optimization strategy for adjusting the step size, default is None.

    Key Methods:
    - _next(x, func_callback, grad_func_callback, hessian_func_callback):
        Computes the next point in the optimization process based on the current point x.
        It begins by evaluating the Hessian at the current point if it's not already defined.
        It then calculates the search direction using the negative of the product of the
        current inverse Hessian and the gradient. After determining the new point, it updates
        the inverse Hessian approximation using the BFGS formula, which takes into account the
        changes in the gradient and position.
    - optimize(x, func_callback, grad_func_callback, hessian_func_callback, is_plot):
        Iteratively updates the point x until convergence is achieved, defined by the norm of the
        gradient falling below a specified epsilon value. If is_plot is True, it records the points
        visited during the optimization process for potential visualization.

    The BFGS algorithm is particularly useful in scenarios where the Hessian matrix is expensive
    to compute directly. It effectively balances computational efficiency with convergence properties,
    making it a preferred choice for many optimization problems.
    """

    def __init__(self, alpha: float = 0.01, alpha_optim: None | Optim = None) -> None:
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        self.hessian_k: ndarray | None = None
        return

    def _reset(self) -> None:
        self.alpha = 0.01
        self.num_iter = 0
        self.hessian_k: ndarray | None = None
        return

    def _next(
        self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback
    ) -> ndarray:

        # first step, take real hessian not approximation
        # this wont run for subsequent iterations
        if not isinstance(self.hessian_k, ndarray):
            self.hessian_k = hessian_func_callback(x)

        assert isinstance(self.hessian_k, ndarray), "hessian not defined correctly"
        assert eig(self.hessian_k)[0].all() > 0, "Not positive definite hessian"

        direction = -self.hessian_k @ grad_func_callback(x)

        x_new = x + self.alpha * direction

        del_x = self.alpha * direction
        del_grad = grad_func_callback(x_new) - grad_func_callback(x)

        # resizing del_x and del_grad to encounter transposing issue
        assert isinstance(del_x, ndarray)
        assert isinstance(del_grad, ndarray)

        del_x.resize((len(del_x), 1))
        del_grad.resize((len(del_grad), 1))

        self.hessian_k = inv(
            inv(self.hessian_k)
            + (del_grad @ del_grad.T) / (del_grad.T @ del_x)
            - (inv(self.hessian_k) @ del_x @ del_x.T @ inv(self.hessian_k))
            / (del_x.T @ inv(self.hessian_k) @ del_x)
        )

        return x_new

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback, hessian_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()

        if is_plot:
            return x, plot_points
        return x


class ConjugateGradient(Optim):
    """
    Implementation of the Conjugate Gradient method for optimization.

    The Conjugate Gradient algorithm is an iterative method for solving systems of linear equations
    with a symmetric positive-definite matrix. It is particularly useful for large-scale optimization
    problems where the Hessian matrix is either too expensive to compute or store.

    This method leverages the properties of conjugate directions to minimize a quadratic objective
    function. It is commonly applied in optimization tasks, especially in scenarios involving
    large datasets or complex models.

    Key Methods:
    - _next(x, func_callback, grad_func_callback, hessian_func_callback):
        Computes the next point in the optimization process based on the current point x.
        The method calculates a step size using the current direction and the gradient of the function,
        then updates the position. It also computes a correlation factor (beta) to update the direction
        for the next iteration, ensuring that the new direction remains conjugate to the previous ones.

    - optimize(x, func_callback, grad_func_callback, hessian_func_callback, is_plot):
        Iteratively updates the point x until convergence is achieved, defined by the norm of the
        gradient falling below a specified epsilon value. The initial direction is set to the negative
        gradient at the starting point. If is_plot is True, it records the points visited during the
        optimization process for potential visualization.

    The Conjugate Gradient method is known for its efficiency in high-dimensional spaces, especially
    when combined with line search techniques to find optimal step sizes for each update.
    """

    def __init__(self) -> None:
        self.direction: ndarray | None = None
        self.beta: float | None = None
        return

    def _reset(self) -> None:
        self.direction = None
        self.beta = None
        self.num_iter = 0
        return

    def _next(
        self, x: ndarray, func_callback, grad_func_callback, hessian_func_callback
    ) -> ndarray:

        # For Line Search
        assert isinstance(
            self.direction, ndarray
        ), "Initial direction vector not defined"
        step_size = -(self.direction.T @ grad_func_callback(x)) / (
            self.direction.T @ hessian_func_callback(x) @ self.direction
        )

        x_new = x + step_size * self.direction

        # calculate correlation factor
        self.beta = -(
            self.direction.T @ hessian_func_callback(x) @ grad_func_callback(x_new)
        ) / (self.direction.T @ hessian_func_callback(x) @ self.direction)

        assert self.beta, "Error in Beta calculation"
        # new conjugate direction
        self.direction = -grad_func_callback(x_new) + self.beta * self.direction

        return x_new

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        plot_points: list[ndarray] = [x]
        self.direction = -grad_func_callback(x)

        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1
            x = self._next(x, func_callback, grad_func_callback, hessian_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x
