import numpy as np
from utils.base import Optim

EPS = 0.3

class GoldenSearch(Optim):
    def __init__(self, phi: float = 0.618) -> None:
        self.phi = phi 

    def _first(self, a, b) -> tuple[float, float]:
        x1 = b - (b - a) * self.phi
        x2 = a + (b - a) * self.phi
        return x1, x2

    def _next_b(self, a, b) -> float:
        return b - (b - a) * self.phi

    def _next_a(self, a, b) -> float:
        return a + (b - a) * self.phi

    def optimize(self, func_callback, lower_bound: float, upper_bound: float) -> float:
        a, b = lower_bound, upper_bound
        x1, x2 = self._first(a, b)

        while abs(b - a) > EPS:
            if func_callback(x1) < func_callback(x2):
                b = x2
                x2 = x1
                x1 = self._next_b(a, b)
            else:
                a = x1
                x1 = x2
                x2 = self._next_a(a, b)

        return (a + b) / 2


class Backtracking(Optim):
    def __init__(self, alpha: float = 1.0, beta: float = 0.5, delta: float = 0.1, isArmijo: bool = True) -> None:
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.isArmijo = isArmijo

    def _check_condition(self, func_callback, func_grad_callback, x, alpha) -> bool:
        if self.isArmijo:
            return func_callback(x - alpha * func_grad_callback(x)) > func_callback(x) - self.delta * alpha * (func_grad_callback(x).T @ func_grad_callback(x))
        else:
            return func_callback(x - alpha * func_grad_callback(x)) > func_callback(x)

    def optimize(self, func_callback, func_grad_callback, x_k) -> float:
        alpha = self.alpha

        while self._check_condition(func_callback, func_grad_callback, x_k, alpha):
            alpha *= self.beta

        return alpha

if __name__ == "__main__":

    # func = lambda x: x**4 - 14 * x**3 + 60 * x**2 - 70 * x
    # low_bound = 0
    # up_bound = 2

    # gs = GoldenSearch()
    # soln = gs.optimize(func, low_bound, up_bound)

    func = lambda x: 4 * x[0] ** 2 + x[1] ** 2 - 2 * x[0] * x[1]
    grad = lambda x: np.array([8 * x[0] - 2 * x[1], 2 * x[1] - 2 * x[0]])

    btls = Backtracking(alpha=1)
    soln = np.array([10, 10])
    while np.linalg.norm(grad(soln)) > 1e-3:
        alpha = btls.optimize(func, grad, soln)
        soln = soln - alpha * grad(soln)
        print(f" -> Alpha = {alpha}, New X = {soln}")

    print("Optimal Solution:", soln)
