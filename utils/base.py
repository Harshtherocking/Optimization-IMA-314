import numpy as np
from numpy import ndarray
from abc import abstractmethod, ABC 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EPS = 1e-6

class Optim (ABC): 
    '''
    Base class for optimizers
    '''
    num_iter : int = 0

    @abstractmethod
    def optimize (self, x : ndarray, func_callback, grad_func_callback, hessian_func_callback, is_plot : bool) -> ndarray | tuple[ndarray,list[ndarray]]:
        pass


class Function : 
    '''
    Base class for functions
    func : function which should return : ndarray() object type
    grad : gradient function returning ndarray() obeject type
    '''
    def __init__ (self, func , grad_func = None, hessian_func = None, name : str = "myFunc" ) -> None :
        self.__func = func
        self.__grad_func = grad_func
        self.__hessian_func = hessian_func
        self.__name = name
        return None

    def __call__ (self, x : ndarray) -> ndarray :  
        return self.__func(x)

    def __repr__(self) -> str:
        return self.__name

    def grad(self, x: np.ndarray) -> np.ndarray:
        if self.__grad_func:
            return self.__grad_func(x)

        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = np.array(x)
            x_minus = np.array(x)
            x_plus[i] += EPS
            x_minus[i] -= EPS
            grad[i] = (self.__func(x_plus) - self.__func(x_minus)) / (2 * EPS)
        return grad

    def hessian(self, x: np.ndarray) -> np.ndarray:
        if self.__hessian_func:
            return self.__hessian_func(x)

        n = len(x)
        hessian_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_plus_plus = np.array(x)
                x_plus_minus = np.array(x)
                x_minus_plus = np.array(x)
                x_minus_minus = np.array(x)
                
                x_plus_plus[i] += EPS
                x_plus_plus[j] += EPS
                x_plus_minus[i] += EPS
                x_minus_minus[j] -= EPS
                x_minus_plus[i] -= EPS

                f_pp = self.__func(x_plus_plus)
                f_pm = self.__func(x_plus_minus)
                f_mp = self.__func(x_minus_plus)
                f_mm = self.__func(x_minus_minus)

                hessian_matrix[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * EPS ** 2)
        return hessian_matrix

    def optimize (self, initial_val : ndarray, optim: Optim, is_plot : bool = False) -> ndarray :
        soln = optim.optimize(initial_val, self.__call__, self.grad, self.hessian, is_plot = is_plot) 

        if is_plot and isinstance(soln, tuple): 
            # plot the trajectory 
            self.plot(points = soln[1])
            assert isinstance(soln[0], ndarray)
            return soln[0]

        assert isinstance(soln, ndarray), "Value recieved from Optim is corrupted"
        return soln

    def plot (self,
              points : list[ndarray] | ndarray | None  = None,
              x_val : tuple[int,int] = (-10,10),
              y_val : tuple[int,int] = (-10,10),
              num_points : int = 100) -> None : 
        '''
        Implemented only for 2-D
        '''

        if points :
            points_array = np.array(points)
            x_val = (np.min(points_array[:, 0]) - 1, np.max(points_array[:, 0]) + 1)
            y_val = (np.min(points_array[:, 1]) - 1, np.max(points_array[:, 1]) + 1)

        x = np.linspace(x_val[0], x_val[1], num_points)
        y = np.linspace(y_val[0], y_val[1], num_points)
        X, Y = np.meshgrid(x, y)

        Z = np.array([self.__call__(np.array([xi, yi])) for xi, yi in zip(X.flatten(), Y.flatten())]).reshape(X.shape)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap='cividis', rstride=1, cstride=1, linewidth=0, antialiased=True, alpha = 0.8)
        # surf = ax.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False, alpha=0.7)

        # scatter plot the trajectory of points 
        if points is not None and len(points) > 0:
                # Extract x, y from each point and calculate Z values
                x_points = np.array([p[0] for p in points])  # X values
                y_points = np.array([p[1] for p in points])  # Y values
                z_points = np.array([self.__call__(p) for p in points])  # Z values

                ax.plot(x_points, y_points, z_points, color='r', marker='o', label='Trajectory', markersize = 5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(self.__repr__())

        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()


if __name__ == "__main__" :
    f = lambda x : x[0]**2 + x[1] **2
    g = lambda x : np.array([2*x[0], 2*x[1]])

    func = Function(f,g, "func")

    x  = np.array([3,8])

    print(func(x))
    print(func.grad(x))
    print(func)



class Algo (): 
    '''
    Base class for algorithms
    '''
    @abstractmethod
    def __init__ (self, optim : Optim, *args, **kwargs) -> None : 
        pass

    @abstractmethod
    def train (self, X_train : ndarray, Y_train : ndarray, epochs : int = 1, is_plot : bool = False) -> None :
        pass

    @abstractmethod
    def test (self, X_test : ndarray, Y_test : ndarray, is_plot : bool) -> None : 
        pass

    @abstractmethod
    def __call__ (self, X : ndarray, is_plot : bool = False) -> ndarray : 
        pass
