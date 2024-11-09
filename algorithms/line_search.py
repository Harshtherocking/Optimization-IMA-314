from algorithms.base import Optim

EPS = 0.3


class GoldenSearch(Optim):
    """
    Implementation of the Golden Search Optimization Algorithm.
    - phi: The golden ratio constant used for the search, default is approximately 0.68103.
    Key Methods:
    - _first(a, b): Calculates the first two points (x1 and x2) to evaluate in the interval [a, b].
    - _next_b(a, b): Computes the next point to evaluate by moving from the right endpoint (b) of the interval.
    - _next_a(a, b): Computes the next point to evaluate by moving from the left endpoint (a) of the interval.
    - optimize(x, Func, func_callback, grad_func_callback, lower_bound, upper_bound):
        Performs the golden search algorithm to find the minimum of a unimodal function in the interval [lower_bound, upper_bound].
        If a custom function Func is not provided, it uses the given func_callback and grad_func_callback to define the target function.
        The optimization process continues until the width of the interval is less than a small threshold (EPS).
    The Golden Search method is useful for efficiently finding the minimum of unimodal functions.
    """

    def __init__(self, phi: float = 0.68103) -> None:
        self.phi = phi

    def _first(self, a, b) -> tuple[float, float]:
        x1 = b - (b - a) * self.phi
        x2 = a + (b - a) * self.phi
        return x1, x2

    def _next_b(self, a, b) -> float:
        return b - (b - a) * self.phi

    def _next_a(self, a, b) -> float:
        return a + (b - a) * self.phi

    def optimize(
        self,
        x,
        Func,
        func_callback,
        grad_func_callback,
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        # defining target function to optimize
        Func = (
            Func
            if Func
            else lambda alpha: func_callback(x - alpha * grad_func_callback(x))
        )

        a, b = lower_bound, upper_bound
        x1, x2 = self._first(a, b)

        while abs(b - a) > EPS:
            if Func(x1) < Func(x2):
                b = x2
                x2 = x1
                x1 = self._next_b(a, b)
            else:
                a = x1
                x1 = x2
                x2 = self._next_a(a, b)

        return (a + b) / 2


class Backtracking(Optim):
    """
    Implementation of the Backtracking Line Search Algorithm for optimization.
    - beta: Diminishing factor to reduce the step size in each iteration, default is 0.5.
    - delta: A controlling factor used to ensure sufficient decrease in the objective function, default is 0.1.
    - isArmijo: A boolean flag indicating whether to use the Armijo condition for the line search, default is True.
    Key Methods:
    - _check_condition(alpha, Func): Checks if the current step size alpha satisfies the condition defined by Func.
    - optimize(x, Func, func_callback, grad_func_callback, lower_bound, upper_bound):
        Performs the backtracking line search to find an appropriate step size alpha that satisfies the specified conditions.
        The optimization process reduces alpha by the factor beta until the condition is no longer met or alpha is less than the lower_bound.
        If isArmijo is True, the Armijo condition is used; otherwise, it falls back to a simple comparison.
    The Backtracking method is effective in adjusting step sizes to improve convergence in gradient-based optimization algorithms.
    """

    def __init__(
        self, beta: float = 0.5, delta: float = 0.1, isArmijo: bool = True
    ) -> None:
        self.beta = beta        # diminishing factor
        self.delta = delta      # controlling factor
        # boolean controller to use default Backtracking Line Search with or without Armijo Rule
        self.isArmijo = isArmijo

    def _check_condition(
        self, x, Func, func_callback, grad_func_callback, alpha
    ) -> bool:
        if self.isArmijo:
            return func_callback(x - alpha * grad_func_callback(x)) > func_callback(x) - self.delta * alpha * (grad_func_callback(x).T @ grad_func_callback(x))
        else:
            if Func:
                return Func(alpha)
            return func_callback(x - alpha * grad_func_callback(x)) > func_callback(x)

    def optimize(
        self,
        x,
        Func,
        func_callback,
        grad_func_callback,
        lower_bound: float,
        upper_bound: float,
    ) -> float:
        # Initializing alpha to Upper Bound
        alpha = upper_bound
        # Checking Normal Backtracking / Armijo Rule Condition
        while (
            self._check_condition(x, Func, func_callback, grad_func_callback, alpha)
            and alpha > lower_bound
        ):
            alpha *= self.beta

        return alpha
