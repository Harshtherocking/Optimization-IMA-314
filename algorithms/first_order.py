import numpy as np
from numpy import ndarray
from algorithms.base import Optim, Function

EPSILON = 0.0001


class GradientDescent(Optim):
    """
    Implementation of the Gradient Descent (I-Order) Algorithm.
    - alpha: The learning rate, controlling the step size for each iteration. The default value is 0.01.
    - alpha_optim: Optional parameter for performing line search to dynamically adjust the learning rate `alpha`.
    Key Methods:
    - _reset(): Resets the learning rate and iteration count after the optimization is done.
    - _next(): Calculates the next point by moving in the opposite direction of the gradient. If `alpha_optim` is set, it performs a line search to optimize the step size.
    - optimize(): Runs the iterative gradient descent process, updating `x` until the gradient's norm is smaller than a predefined threshold (EPSILON). Can optionally return a list of points for plotting.
    This implementation allows basic gradient descent and includes support for line search to adjust the learning rate dynamically during the optimization process.
    """

    def __init__(self, alpha: float = 0.01, alpha_optim: None | Optim = None) -> None:
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        return

    def _reset(self) -> None:
        self.alpha = 0.01
        self.num_iter = 0
        return

    def _next(self, x: ndarray, func_callback, grad_func_callback) -> ndarray:

        # For Line Search
        if isinstance(self.alpha_optim, Optim):
            self.alpha = self.alpha_optim.optimize(
                x=x,
                Func=lambda alpha: func_callback(x - alpha * grad_func_callback(x)),
                func_callback=func_callback,
                grad_func_callback=grad_func_callback,
                lower_bound=0,
                upper_bound=1,
            )

        return x - self.alpha * grad_func_callback(x)

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
            # Position Update Equation of the Gradient Descent Algorithm: x = x - alpha * grad(x)
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x


class MomentumGradientDescent(Optim):
    """
    Implementation of Momentum Gradient Descent optimization.

    This optimizer enhances gradient descent by incorporating a momentum term,
    which helps accelerate convergence in the relevant direction and reduce oscillations.
    The momentum is updated iteratively along with the learning rate (alpha).

    Key Methods:
    - optimize(x, func_callback, grad_func_callback, hessian_func_callback, is_plot=False):
        Optimizes the input parameters using gradient descent with momentum. Optionally returns
        the points traversed during optimization for plotting.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        alpha_optim: Optim | None = None,
        momentum_coeff: float = 0.75,
    ) -> None:
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        self.momentum_coeff = momentum_coeff
        self.momentum: ndarray | None = None
        return

    def _reset(self) -> None:
        self.alpha = 0.01
        self.momentum_coeff = 0.75
        self.momentum: ndarray | None = None
        self.num_iter = 0
        return

    def _next(self, x: ndarray, func_callback, grad_func_callback) -> ndarray:
        assert isinstance(self.momentum, ndarray), "initial momentum not defined"

        # For Line Search
        if isinstance(self.alpha_optim, Optim):
            self.alpha = self.alpha_optim.optimize(
                x=x,
                Func=lambda alpha: func_callback(x - self.momentum_coeff * self.momentum + alpha * grad_func_callback(x)),
                func_callback=func_callback,
                grad_func_callback=grad_func_callback,
                lower_bound=0,
                upper_bound=1,
            )

        self.momentum = (
            self.momentum_coeff * self.momentum + self.alpha * grad_func_callback(x)
        )
        return x - self.momentum

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        self.momentum = np.zeros(x.shape)
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1

            # Position Update Equation of the Momentum Algorithm: x = x - (alpha * grad(x) + gamma * momentum)
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x


class NesterovAcceleratedGradientDescent(Optim):
    """
    Implementation of the Nesterov Accelerated Gradient Descent or NGD (I-Order) Algorithm.
    - alpha: The learning rate for each iteration (default is 0.01).
    - alpha_optim: Optional parameter for line search to dynamically adjust `alpha`.
    - momentum_coff: Coefficient controlling the contribution of momentum (default is 0.75).
    Key Methods:
    - _reset(): Resets the learning rate, momentum coefficient, and momentum vector at the end of optimization.
    - _next(): Updates the position using the NAG formula. First, a look-ahead position is calculated using momentum, and then the gradient is applied.
    - optimize(): Iteratively updates `x` using NAG until the norm of the gradient is smaller than a threshold (EPSILON). Initializes momentum and can optionally return a list of points for plotting.
    This implementation enhances gradient descent by incorporating a look-ahead mechanism, helping the algorithm converge faster using momentum.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        alpha_optim: Optim | None = None,
        momentum_coeff: float = 0.75,
    ) -> None:
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        self.momentum_coeff = momentum_coeff
        self.momentum: ndarray | None = None
        return

    def _reset(self) -> None:
        self.alpha = 0.01
        self.momentum_coeff = 0.75
        self.momentum: ndarray | None = None
        self.num_iter = 0
        return

    def _next(self, x: ndarray, func_callback, grad_func_callback) -> ndarray:
        assert isinstance(self.momentum, ndarray), "initial momentum not defined"

        x_look_ahead = x - self.momentum_coeff * self.momentum

        # For Line Search
        if isinstance(self.alpha_optim, Optim):
            self.alpha = self.alpha_optim.optimize(
                x=x,
                Func=lambda alpha: func_callback(x - self.momentum_coeff * self.momentum + alpha * grad_func_callback(x_look_ahead)),
                func_callback=func_callback,
                grad_func_callback=grad_func_callback,
                lower_bound=0,
                upper_bound=1,
            )

        self.momentum = (
            self.momentum_coeff * self.momentum
            + self.alpha * grad_func_callback(x_look_ahead)
        )

        return x - self.momentum

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        self.momentum = np.zeros(x.shape)
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1

            # Position Update Equation of the NGD Algorithm: x = x - (alpha * grad(x_look_ahead) + gamma * momentum)
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x


class Adagrad(Optim):
    """
    Implementation of the Adagrad (I-Order) Algorithm.
    - alpha: The learning rate for each iteration (default is 0.01).
    - alpha_optim: Optional parameter for line search to dynamically adjust `alpha`.
    Key Methods:
    - _reset(): Resets the square gradient accumulation and the iteration counter at the end of optimization.
    - _next(): Updates the position using the Adagrad formula. It accumulates the squared gradients and adjusts the learning rate based on this accumulation.
    - optimize(): Iteratively updates `x` using Adagrad until the norm of the gradient is smaller than a threshold (EPSILON). Initializes the squared gradient accumulator and can optionally return a list of points for plotting.
    This implementation adapts the learning rate based on the history of gradients, improving convergence for sparse data.
    """

    def __init__(self, alpha: float = 0.01, alpha_optim: Optim | None = None) -> None:
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        self.sq_grad_acc: ndarray | None = None
        return

    def _reset(self) -> None:
        self.sq_grad_acc: ndarray | None = None
        self.num_iter = 0
        return

    def _next(self, x: ndarray, func_callback, grad_func_callback) -> ndarray:
        assert isinstance(
            self.sq_grad_acc, ndarray
        ), "initial square gradient accumulation not defined"

        self.sq_grad_acc += np.square(grad_func_callback(x))

        # For Line Search
        if isinstance(self.alpha_optim, Optim):
            self.alpha = self.alpha_optim.optimize(
                x=x,
                Func=lambda alpha: func_callback(x - alpha / np.sqrt(self.sq_grad_acc + EPSILON) * grad_func_callback(x)),
                func_callback=func_callback,
                grad_func_callback=grad_func_callback,
                lower_bound=0,
                upper_bound=1,
            )

        assert isinstance(self.sq_grad_acc, ndarray), "problem in accumulation"
        return x - self.alpha / np.sqrt(
            self.sq_grad_acc + EPSILON
        ) * grad_func_callback(x)

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        self.sq_grad_acc = np.zeros(x.shape)
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1
            # Position Update Equation of the Adagrad Algorithm: x = x - alpha / sqrt(sq_grad_acc + epsilon) * grad(x)
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x


class RMSProp(Optim):
    """
    Implementation of the RMSProp (I-Order) Algorithm.
    - alpha: The learning rate for each iteration (default is 0.01).
    - alpha_optim: Optional parameter for line search to dynamically adjust `alpha`.
    - beta: Coefficient controlling the decay rate of the squared gradient (default is 0.9).
    Key Methods:
    - _reset(): Resets the square gradient accumulation and the iteration counter at the end of optimization.
    - _next(): Updates the position using the RMSProp formula. It accumulates the squared gradients using an exponential decay factor and adjusts the learning rate based on this accumulation.
    - optimize(): Iteratively updates `x` using RMSProp until the norm of the gradient is smaller than a threshold (EPSILON). Initializes the squared gradient accumulator and can optionally return a list of points for plotting.
    This implementation helps stabilize the learning rate by scaling it based on recent gradient magnitudes.
    """

    def __init__(
        self, alpha: float = 0.01, alpha_optim: Optim | None = None, beta: float = 0.9
    ) -> None:
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        self.beta = beta
        self.sq_grad_acc: ndarray | None = None
        return

    def _reset(self) -> None:
        self.sq_grad_acc: ndarray | None = None
        self.num_iter = 0
        return

    def _next(self, x: ndarray, func_callback, grad_func_callback) -> ndarray:
        assert isinstance(
            self.sq_grad_acc, ndarray
        ), "initial square gradient accumulation not defined"

        self.sq_grad_acc = self.beta * self.sq_grad_acc + (1 - self.beta) * np.square(
            grad_func_callback(x)
        )

        # For Line Search
        if isinstance(self.alpha_optim, Optim):
            self.alpha = self.alpha_optim.optimize(
                x=x,
                Func=lambda alpha: func_callback(x - alpha / np.sqrt(self.sq_grad_acc + EPSILON) * grad_func_callback(x)),
                func_callback=func_callback,
                grad_func_callback=grad_func_callback,
                lower_bound=0,
                upper_bound=1,
            )

        assert isinstance(self.sq_grad_acc, ndarray), "problem in accumulation"
        return x - self.alpha / np.sqrt(
            self.sq_grad_acc + EPSILON
        ) * grad_func_callback(x)

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        self.sq_grad_acc = np.zeros(x.shape)
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1
            # Position Update Equation of the RMSProp Algorithm: x = x - alpha / sqrt(sq_grad_acc + epsilon) * grad(x)
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x


class Adam(Optim):
    """
    Implementation of the Adam (I-Order) Optimization Algorithm.
    - alpha: The learning rate for each iteration (default is 0.01).
    - alpha_optim: Optional parameter for line search to dynamically adjust `alpha`.
    - beta_1: Coefficient controlling the decay rate of the first moment estimates (default is 0.9).
    - beta_2: Coefficient controlling the decay rate of the second moment estimates (default is 0.99).
    Key Methods:
    - _reset(): Resets the first and second moment accumulators and the iteration counter at the end of optimization.
    - _next(): Updates the position using the Adam formula. It computes the first and second moment estimates and applies bias correction to adjust the learning rates.
    - optimize(): Iteratively updates `x` using Adam until the norm of the gradient is smaller than a threshold (EPSILON). Initializes the moment accumulators and can optionally return a list of points for plotting.
    This implementation combines the advantages of both AdaGrad and RMSProp to provide an effective optimization algorithm for various machine learning tasks.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        alpha_optim: Optim | None = None,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
    ) -> None:
        self.alpha = alpha
        self.alpha_optim = alpha_optim
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.first_moment_acc: ndarray | None = None
        self.second_moment_acc: ndarray | None = None
        return

    def _reset(self) -> None:
        self.first_moment_acc: ndarray | None = None
        self.second_moment_acc: ndarray | None = None
        self.num_iter = 0
        return

    def _next(self, x: ndarray, func_callback, grad_func_callback) -> ndarray:
        assert isinstance(
            self.first_moment_acc, ndarray
        ), "initial first order accumulation not defined"
        assert isinstance(
            self.second_moment_acc, ndarray
        ), "initial second order accumulation not defined"

        self.first_moment_acc = self.beta_1 * self.first_moment_acc + (
            1 - self.beta_1
        ) * grad_func_callback(x)
        self.second_moment_acc = self.beta_2 * self.second_moment_acc + (
            1 - self.beta_2
        ) * np.square(func(x))


        assert isinstance(
            self.first_moment_acc, ndarray
        ), "problem in first order accumulation"
        assert isinstance(
            self.second_moment_acc, ndarray
        ), "problem in second order accumulation"

        first_order_corrected = self.first_moment_acc / (1 - self.beta_1)
        second_order_corrected = self.second_moment_acc / (1 - self.beta_2)

        assert isinstance(
            first_order_corrected, ndarray
        ), "problem in first order correction"
        assert isinstance(
            second_order_corrected, ndarray
        ), "problem in second order correction"

        # For Line Search
        if isinstance(self.alpha_optim, Optim):
            self.alpha = self.alpha_optim.optimize(
                x=x,
                Func=lambda alpha: func_callback(x - alpha / np.sqrt(second_order_corrected + EPSILON) * first_order_corrected),
                func_callback=func_callback,
                grad_func_callback=grad_func_callback,
                lower_bound=0,
                upper_bound=1,
            )

        return (
            x
            - self.alpha
            / np.sqrt(second_order_corrected + EPSILON)
            * first_order_corrected
        )

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        self.first_moment_acc = np.zeros(x.shape)
        self.second_moment_acc = np.zeros(x.shape)
        plot_points: list[ndarray] = [x]

        while np.linalg.norm(grad_func_callback(x)) > EPSILON:
            self.num_iter += 1
            # Position Update Equation of the Adam Algorithm: x = x - alpha / sqrt(1 - beta_2 ^ t) / (1 - beta_1 ^ t) * grad(x)
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x


class Subgradient(Optim):
    """
    Implementation of the Subgradient Optimization (I-Order w. Continuous & Non-Differentiable) Algorithm.
    - alpha: The learning rate for each iteration (default is 0.01).
    - f_best: Holds the best function value found during optimization.
    - K: A counter for maximum iterations (default is 100).
    Key Methods:
    - _reset(): Resets the learning rate, counter K, and iteration count at the end of optimization.
    - _next(): Updates the position using the subgradient method. It computes the new position based on the gradient and updates f_best if the new function value is better.
    - optimize(): Iteratively updates `x` using the subgradient method until the norm of the gradient is smaller than a threshold (EPSILON) or the maximum iterations (K) are reached. Initializes the best function value and can optionally return a list of points for plotting.
    This algorithm is useful for minimizing non-differentiable convex functions, making it a versatile choice for various optimization problems.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self.f_best: float | None = None
        self.K = 100
        return

    def _reset(self) -> None:
        self.alpha = 0.01
        self.K = 100
        self.num_iter = 0
        return

    def _next(self, x: ndarray, func_callback, grad_func_callback) -> ndarray:
        x_new = x - self.alpha * grad_func_callback(x)

        if func_callback(x_new) < self.f_best:
            self.f_best = func_callback(x_new)
            return x_new

        self.K -= 1
        return x

    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        plot_points: list[ndarray] = [x]
        self.f_best = func_callback(x)

        while np.linalg.norm(grad_func_callback(x)) > EPSILON and self.K > 0:
            self.num_iter += 1
            # Position Update Equation of the Subgradient Algorithm: x = x - alpha * sub_grad(x)
            x = self._next(x, func_callback, grad_func_callback)

            if is_plot:
                plot_points.append(x)

        self._reset()
        if is_plot:
            return x, plot_points
        return x


if __name__ == "__main__":
    f = lambda x: x[0] ** 2 + x[1] ** 2
    g = lambda x: np.array([2 * x[0], 2 * x[1]])

    func = Function(f, g, "func")
    x = np.array([3, 8])

    gd = GradientDescent()

    soln = func.optimize(x, gd)

    print(soln)
