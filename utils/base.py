import numpy as np
from numpy import ndarray
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

EPS = 1e-6


class Optim(ABC):
    """
    Base class for Optimizers
    """

    num_iter: int = 0

    @abstractmethod
    def optimize(
        self,
        x: ndarray,
        func_callback,
        grad_func_callback,
        hessian_func_callback,
        is_plot: bool,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
        pass


class Function:
    """
    Base class for Functions
    func : function which should return : ndarray() object type

    Gradient and Hessian calculation are implemeneted for 2-D inputs.
    For N dimension input, also provide  :
        grad_func : gradient function returning : ndarray() object type
        hessian_func : hessian function returning : ndarray() object type
    """

    def __init__(
        self, func, grad_func=None, hessian_func=None, name: str = "myFunc"
    ) -> None:
        self.__func = func
        self.__grad_func = grad_func
        self.__hessian_func = hessian_func
        self.__name = name
        return None

    def __call__(self, x: ndarray) -> ndarray:
        return self.__func(x)

    def __repr__(self) -> str:
        return self.__name

    def grad(self, x: np.ndarray) -> np.ndarray:
        if self.__grad_func:
            return self.__grad_func(x)

        _x, _y = x
        df_dx = (
            self.__func(np.array([_x + EPS, _y]))
            - self.__func(np.array([_x - EPS, _y]))
        ) / (2 * EPS)
        df_dy = (
            self.__func(np.array([_x, _y + EPS]))
            - self.__func(np.array([_x, _y - EPS]))
        ) / (2 * EPS)
        return np.array([df_dx, df_dy])

    def hessian(self, x: np.ndarray) -> np.ndarray:
        if self.__hessian_func:
            return self.__hessian_func(x)

        _x, _y = x
        d2f_dx2 = (
            self.__func(np.array([_x + EPS, _y]))
            - 2 * self.__func(np.array([_x, _y]))
            + self.__func(np.array([_x - EPS, _y]))
        ) / (EPS**2)
        d2f_dy2 = (
            self.__func(np.array([_x, _y + EPS]))
            - 2 * self.__func(np.array([_x, _y]))
            + self.__func(np.array([_x, _y - EPS]))
        ) / (EPS**2)

        d2f_dxdy = (
            self.__func(np.array([_x + EPS, _y + EPS]))
            - self.__func(np.array([_x + EPS, _y - EPS]))
            - self.__func(np.array([_x - EPS, _y + EPS]))
            + self.__func(np.array([_x - EPS, _y - EPS]))
        ) / (4 * EPS**2)

        hessian_matrix = np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])
        return hessian_matrix

    def optimize(
        self, initial_val: ndarray, optim: Optim, is_plot: bool = False
    ) -> ndarray:
        soln = optim.optimize(
            initial_val, self.__call__, self.grad, self.hessian, is_plot=is_plot
        )

        if is_plot and isinstance(soln, tuple):
            # plot the trajectory
            self.plot(points=soln[1])
            assert isinstance(soln[0], ndarray)
            return soln[0]

        assert isinstance(soln, ndarray), "Value recieved from Optim is corrupted"
        return soln

    def plot(
        self,
        points: list[ndarray] | ndarray | None = None,
        x_range: tuple[int, int] = (-10, 10),
        y_range: tuple[int, int] = (-10, 10),
        num_points: int = 100,
        show: bool = True,
    ) -> None | matplotlib.figure.Figure:

        if points:
            points_array = np.array(points)
            x_range = (np.min(points_array[:, 0]) - 1, np.max(points_array[:, 0]) + 1)
            y_range = (np.min(points_array[:, 1]) - 1, np.max(points_array[:, 1]) + 1)

        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(x, y)

        Z = np.array(
            [
                self.__call__(np.array([xi, yi]))
                for xi, yi in zip(X.flatten(), Y.flatten())
            ]
        ).reshape(X.shape)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="cividis",
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=True,
            alpha=0.8,
        )
        # surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.7)

        # scatter plot the trajectory of points
        if points is not None and len(points) > 0:
            # Extract x, y from each point and calculate Z values
            x_points = np.array([p[0] for p in points])  # X values
            y_points = np.array([p[1] for p in points])  # Y values
            z_points = np.array([self.__call__(p) for p in points])  # Z values

            ax.plot(
                x_points,
                y_points,
                z_points,
                color="r",
                marker="o",
                label="Trajectory",
                markersize=5,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(self.__repr__())

        fig.colorbar(surf, shrink=0.5, aspect=5)
        if show:
            plt.show()
        else:
            return fig


class Algo:
    """
    Base class for the Optimization Algorithms
    """

    @abstractmethod
    def __init__(self, optim: Optim, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def train(
        self, X_train: ndarray, Y_train: ndarray, epochs: int = 1, is_plot: bool = False
    ) -> None:
        pass

    @abstractmethod
    def test(self, X_test: ndarray, Y_test: ndarray, is_plot: bool) -> np.float32:
        pass

    @abstractmethod
    def __call__(self, X: ndarray, is_plot: bool = False) -> ndarray:
        pass
